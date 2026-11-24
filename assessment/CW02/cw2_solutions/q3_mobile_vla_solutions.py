"""Instructor reference solutions for COMP0037 2026/27 Coursework 2 — Question 3.

Implements Week 08 Track A (nanoVLM + Conditional Diffusion Action Head + Chunk-MPC)
on `MobilePickPlaceEnv` (`PandaOmron`):
  - Patches `PixelShuffleProjector.forward`, `build_prefix_causal_mask`,
    `cosine_beta_schedule`, and `evaluate_chunk_mpc` in `cw2_starter/q3_mobile_vla.py`.
  - Trains `NanoVLM` from scratch on 64x64 `agentview` images + 3 mobile-manipulation QA pairs.
  - Trains `ConditionalDiffusionHead` (DDPM on 4-step x 6 active DoF = 24-D chunks) comparing:
      (a) Undertraining (30 epochs) vs. Converged (180 epochs)
      (b) Proprio-Only vs. Proprio + Frozen VLM Embedding
      (c) Receding-Horizon Chunk-MPC (k=1 vs. k=4)
  - Saves:
      - `outputs/q3_vla_training.png`
      - `outputs/q3_chunk_mpc.png`
      - `outputs/q3_vla_rollout.gif` (closed-loop rollout of VLM + Diffusion policy)

Usage (from repository root):
    .venv_robot\\Scripts\\python assessment\\CW02\\cw2_solutions\\q3_mobile_vla_solutions.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
STARTER_DIR = os.path.normpath(os.path.join(CUR_DIR, os.pardir, "cw2_starter"))
OUT_DIR = os.path.join(CUR_DIR, "outputs")
sys.path.insert(0, STARTER_DIR)

import mobile_pick_place_env as mpe  # noqa: E402
import q3_mobile_vla as q3  # noqa: E402

# Active 6 DoF of PandaOmron for this task: arm (dx, dy, dz), base (vx, vy), gripper
ACTIVE_IDX = [0, 1, 2, 7, 8, 10]


def pack_active(actions_11: np.ndarray) -> np.ndarray:
    return actions_11[..., ACTIVE_IDX]


def unpack_active(actions_6: np.ndarray) -> np.ndarray:
    out = np.zeros(actions_6.shape[:-1] + (11,), dtype=np.float32)
    out[..., ACTIVE_IDX] = np.clip(actions_6, -1.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Reference patches for Q3.b (nanoVLM) & Q3.c (DDPM Diffusion + Chunk-MPC)
# --------------------------------------------------------------------------- #

def pixel_shuffle_forward(self, tokens: torch.Tensor) -> torch.Tensor:
    B, N, D = tokens.shape
    g = self.grid
    x = tokens.view(B, g, g, D)
    x = x.view(B, g // 2, 2, g // 2, 2, D).permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, (g // 2) * (g // 2), 4 * D)
    return self.proj(x)


def build_prefix_causal_mask(n_prefix: int, n_text: int, device: torch.device) -> torch.Tensor:
    total = n_prefix + n_text
    mask = torch.zeros((total, total), dtype=torch.float32, device=device)
    causal = torch.triu(torch.full((n_text, n_text), float("-inf"), device=device), diagonal=1)
    mask[n_prefix:, n_prefix:] = causal
    mask[:n_prefix, n_prefix:] = float("-inf")
    return mask


def cosine_beta_schedule(timesteps: int = 50, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    ac = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1.0 - (ac[1:] / ac[:-1])
    return torch.clamp(betas, 1e-5, 0.999).float()


class SimpleVocab:
    def __init__(self, texts: list[str]):
        toks = ["<pad>", "<bos>", "<eos>", "<sep>"]
        seen = set(toks)
        for t in texts:
            for w in t.lower().split():
                if w not in seen:
                    seen.add(w)
                    toks.append(w)
        self.itos = toks
        self.stoi = {w: i for i, w in enumerate(toks)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[w] for w in s.lower().split() if w in self.stoi]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            w = self.itos[i]
            if w == "<eos>":
                break
            if w not in ("<pad>", "<bos>", "<sep>"):
                out.append(w)
        return " ".join(out)


class NanoVLM(nn.Module):
    def __init__(self, vocab_size: int, vit_dim: int = 128, lm_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.vit = q3.TinyViT(img_size=64, patch_size=8, dim=vit_dim, depth=2, heads=4)
        self.projector = q3.PixelShuffleProjector(vit_dim=vit_dim, lm_dim=lm_dim, grid=8)
        self.tok_emb = nn.Embedding(vocab_size, lm_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, lm_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=lm_dim, nhead=4, dim_feedforward=lm_dim * 2, batch_first=True)
        self.lm = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(lm_dim, vocab_size)

    def encode_image(self, img_bchw: torch.Tensor) -> torch.Tensor:
        return self.projector(self.vit(img_bchw))

    def forward_loss(self, img_bchw: torch.Tensor, seq_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        prefix = self.encode_image(img_bchw)
        n_p = prefix.shape[1]
        text_in = seq_ids[:, :-1]
        text_tgt = seq_ids[:, 1:]
        lm_mask = loss_mask[:, 1:]
        B, L = text_in.shape
        txt_emb = self.tok_emb(text_in) + self.pos_emb[:, :L, :]
        x = torch.cat([prefix, txt_emb], dim=1)
        attn_mask = q3.build_prefix_causal_mask(n_p, L, x.device)
        h = self.lm(x, mask=attn_mask)[:, n_p:, :]
        logits = self.head(h)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), text_tgt.reshape(-1), reduction="none").view(B, L)
        return (ce * lm_mask).sum() / (lm_mask.sum() + 1e-8)

    @torch.no_grad()
    def generate(self, img_1chw: torch.Tensor, prompt_ids: list[int], eos_id: int, max_new: int = 8) -> list[int]:
        prefix = self.encode_image(img_1chw)
        n_p = prefix.shape[1]
        ids = list(prompt_ids)
        gen = []
        for _ in range(max_new):
            t_in = torch.tensor([ids], dtype=torch.long, device=img_1chw.device)
            L = t_in.shape[1]
            x = torch.cat([prefix, self.tok_emb(t_in) + self.pos_emb[:, :L, :]], dim=1)
            attn_mask = q3.build_prefix_causal_mask(n_p, L, x.device)
            h = self.lm(x, mask=attn_mask)
            nxt = int(self.head(h[0, -1]).argmax().item())
            if nxt == eos_id:
                break
            ids.append(nxt)
            gen.append(nxt)
        return gen


def train_nanovlm(episodes: list[dict], epochs: int = 12, seed: int = 0, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_train_eps = max(1, len(episodes) - 1)
    all_texts = [t for ep in episodes for step_qas in ep["qa"] for qa in step_qas for t in qa]
    vocab = SimpleVocab(all_texts)

    def _build_samples(eps_subset):
        imgs, seqs, masks, metas = [], [], [], []
        max_len = 0
        raw = []
        for ep in eps_subset:
            for t in range(0, len(ep["images"]), 2):
                for q_txt, a_txt in ep["qa"][t]:
                    p_ids = [vocab.stoi["<bos>"]] + vocab.encode(q_txt) + [vocab.stoi["<sep>"]]
                    a_ids = vocab.encode(a_txt) + [vocab.stoi["<eos>"]]
                    full = p_ids + a_ids
                    max_len = max(max_len, len(full))
                    raw.append((ep["images"][t], p_ids, full, q_txt, a_txt))
        for im, p_ids, full, q_txt, a_txt in raw:
            pad = max_len - len(full)
            seqs.append(full + [0] * pad)
            m = [0.0] * len(p_ids) + [1.0] * (len(full) - len(p_ids)) + [0.0] * pad
            masks.append(m)
            imgs.append(im)
            metas.append((p_ids, q_txt, a_txt))
        return np.asarray(imgs), np.asarray(seqs, dtype=np.int64), np.asarray(masks, dtype=np.float32), metas

    tr_img, tr_seq, tr_msk, _ = _build_samples(episodes[:n_train_eps])
    va_img, va_seq, va_msk, va_meta = _build_samples(episodes[n_train_eps:])

    vlm = NanoVLM(len(vocab.itos)).to(device)
    opt = torch.optim.AdamW(vlm.parameters(), lr=2e-3, weight_decay=1e-4)
    tr_hist, va_hist = [], []
    bs = 64

    for _ in range(epochs):
        vlm.train()
        perm = rng.permutation(len(tr_img))
        ep_l, nb = 0.0, 0
        for i in range(0, len(tr_img), bs):
            idx = perm[i : i + bs]
            im_t = torch.as_tensor(tr_img[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
            sq_t = torch.as_tensor(tr_seq[idx], dtype=torch.long, device=device)
            mk_t = torch.as_tensor(tr_msk[idx], dtype=torch.float32, device=device)
            loss = vlm.forward_loss(im_t, sq_t, mk_t)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vlm.parameters(), 1.0)
            opt.step()
            ep_l += float(loss.item())
            nb += 1
        tr_hist.append(ep_l / max(1, nb))

        vlm.eval()
        with torch.no_grad():
            im_v = torch.as_tensor(va_img[:128], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
            sq_v = torch.as_tensor(va_seq[:128], dtype=torch.long, device=device)
            mk_v = torch.as_tensor(va_msk[:128], dtype=torch.float32, device=device)
            va_hist.append(float(vlm.forward_loss(im_v, sq_v, mk_v).item()))

    vlm.eval()
    correct, total = 0, 0
    by_q_ans: dict[str, Counter] = {}
    for idx in range(min(90, len(va_img))):
        p_ids, q_txt, a_txt = va_meta[idx]
        im_1 = torch.as_tensor(va_img[idx : idx + 1], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        pred_ids = vlm.generate(im_1, p_ids, vocab.stoi["<eos>"])
        correct += int(vocab.decode(pred_ids) == a_txt)
        total += 1
        by_q_ans.setdefault(q_txt, Counter())[a_txt] += 1
    maj_correct = sum(c.most_common(1)[0][1] for c in by_q_ans.values())

    return {
        "model": vlm,
        "vocab": vocab,
        "train_ce": np.asarray(tr_hist),
        "val_ce": np.asarray(va_hist),
        "vqa_acc": correct / max(1, total),
        "maj_acc": maj_correct / max(1, total),
    }


class ConditionalDiffusionHead(nn.Module):
    """DDPM epsilon-predictor on flattened (chunk_len * 6 active DoF = 24-D) action chunks."""

    def __init__(self, proprio_dim: int = 16, vlm_dim: int = 0, chunk_dim: int = 24, timesteps: int = 25):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.vlm_dim = vlm_dim
        self.chunk_dim = chunk_dim
        self.timesteps = timesteps
        betas = q3.cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", ac)
        self.register_buffer("sqrt_ac", torch.sqrt(ac))
        self.register_buffer("sqrt_om_ac", torch.sqrt(1.0 - ac))
        self.register_buffer("prop_mean", torch.zeros(proprio_dim))
        self.register_buffer("prop_std", torch.ones(proprio_dim))

        cond_width = proprio_dim
        if vlm_dim > 0:
            self.vlm_proj = nn.Sequential(nn.LayerNorm(vlm_dim), nn.Linear(vlm_dim, 16), nn.Tanh())
            nn.init.normal_(self.vlm_proj[1].weight, std=0.05)
            nn.init.zeros_(self.vlm_proj[1].bias)
            cond_width += 16
        else:
            self.vlm_proj = None

        self.t_emb = nn.Sequential(nn.Linear(16, 64), nn.SiLU(), nn.Linear(64, 64))
        self.net = nn.Sequential(
            nn.Linear(chunk_dim + cond_width + 64, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, chunk_dim),
        )

    def _sin_time(self, t_idx: torch.Tensor) -> torch.Tensor:
        half = 8
        freqs = torch.exp(torch.linspace(0, math.log(1000.0), half, device=t_idx.device) * -1.0)
        ang = t_idx.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, x_noisy: torch.Tensor, t_idx: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        pn = (cond[:, : self.proprio_dim] - self.prop_mean) / (self.prop_std + 1e-6)
        if self.vlm_proj is not None:
            vf = self.vlm_proj(cond[:, self.proprio_dim :])
            c = torch.cat([pn, vf], dim=-1)
        else:
            c = pn
        te = self.t_emb(self._sin_time(t_idx))
        return self.net(torch.cat([x_noisy, c, te], dim=-1))

    def q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        s1 = self.sqrt_ac[t_idx].unsqueeze(-1)
        s2 = self.sqrt_om_ac[t_idx].unsqueeze(-1)
        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        B = cond.shape[0]
        x = torch.randn((B, self.chunk_dim), device=cond.device)
        for t in reversed(range(self.timesteps)):
            t_idx = torch.full((B,), t, dtype=torch.long, device=cond.device)
            eps_hat = self.forward(x, t_idx, cond)
            ac_t = self.alpha_cumprod[t]
            a_t = self.alphas[t]
            b_t = self.betas[t]
            x0_hat = torch.clamp((x - self.sqrt_om_ac[t] * eps_hat) / self.sqrt_ac[t].clamp(min=1e-4), -1.0, 1.0)
            if t > 0:
                ac_prev = self.alpha_cumprod[t - 1]
                c1 = (torch.sqrt(ac_prev) * b_t) / (1.0 - ac_t)
                c2 = (torch.sqrt(a_t) * (1.0 - ac_prev)) / (1.0 - ac_t)
                var = b_t * (1.0 - ac_prev) / (1.0 - ac_t)
                x = c1 * x0_hat + c2 * x + torch.sqrt(var.clamp(min=1e-8)) * torch.randn_like(x)
            else:
                x = x0_hat
        return torch.clamp(x, -1.0, 1.0)


def train_diffusion_policy(
    imgs: np.ndarray,
    proprios: np.ndarray,
    chunks_11: np.ndarray,
    vlm: NanoVLM | None = None,
    epochs: int = 180,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Pack 4-step x 11-D -> 4-step x 6 active DoF = 24-D
    chunks_6 = pack_active(chunks_11.reshape(-1, 4, 11)).reshape(-1, 24)

    if vlm is not None:
        vlm.eval()
        embs = []
        with torch.no_grad():
            for i in range(0, len(imgs), 64):
                im_t = torch.as_tensor(imgs[i : i + 64], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
                embs.append(vlm.encode_image(im_t).mean(dim=1).cpu().numpy())
        cond_all = np.concatenate([proprios, np.concatenate(embs, axis=0)], axis=1)
    else:
        cond_all = proprios

    vlm_dim = cond_all.shape[1] - proprios.shape[1]
    head = ConditionalDiffusionHead(proprio_dim=proprios.shape[1], vlm_dim=vlm_dim, chunk_dim=24, timesteps=25).to(device)
    head.prop_mean.copy_(torch.as_tensor(proprios.mean(axis=0), dtype=torch.float32, device=device))
    head.prop_std.copy_(torch.as_tensor(proprios.std(axis=0) + 1e-6, dtype=torch.float32, device=device))

    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    c_t = torch.as_tensor(cond_all, dtype=torch.float32, device=device)
    x_t = torch.as_tensor(chunks_6, dtype=torch.float32, device=device)
    losses = []
    n = len(cond_all)

    for _ in range(epochs):
        perm = rng.permutation(n)
        ep_l, nb = 0.0, 0
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            x0 = x_t[idx]
            cb = c_t[idx]
            t_idx = torch.randint(0, head.timesteps, (len(idx),), device=device)
            noise = torch.randn_like(x0)
            x_noisy = head.q_sample(x0, t_idx, noise)
            pred = head(x_noisy, t_idx, cb)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            ep_l += float(loss.item())
            nb += 1
        sched.step()
        losses.append(ep_l / max(1, nb))

    return {"head": head, "losses": np.asarray(losses)}


def evaluate_chunk_mpc(
    policy_fn,
    n_episodes: int = 4,
    chunk_len: int = 4,
    exec_steps: int = 1,
    horizon: int = 240,
    seed: int = 100,
    use_camera: bool = True,
    record_gif_path: str | None = None,
) -> dict:
    need_cam = bool(use_camera or record_gif_path)
    cam_size = 192 if record_gif_path else 64
    env = mpe.make_mobile_pick_place_env(
        use_camera_obs=need_cam, camera_names="agentview", camera_size=cam_size, horizon=horizon, seed=seed
    )
    succ, dists, steps_l = [], [], []
    best_frames: list[np.ndarray] = []
    for ep in range(n_episodes):
        obs = env.reset()
        ep_frames = [np.flipud(obs["agentview_image"]).copy()] if record_gif_path else []
        done_ok = False
        t = 0
        while t < horizon:
            chunk = policy_fn(obs).reshape(chunk_len, 11)
            for k in range(min(exec_steps, chunk_len)):
                obs, _, done, _ = env.step(chunk[k])
                t += 1
                if record_gif_path and (t % 2 == 0 or env._check_success()):
                    ep_frames.append(np.flipud(obs["agentview_image"]).copy())
                if env._check_success():
                    done_ok = True
                    break
                if done or t >= horizon:
                    break
            if done_ok or t >= horizon:
                break
        succ.append(float(done_ok))
        dists.append(float(np.linalg.norm(obs["cube_pos"][:2] - mpe.TARGET_PAD_POS[:2])))
        steps_l.append(t)
        if record_gif_path and (done_ok or not best_frames):
            best_frames = ep_frames
    env.close()

    if record_gif_path and best_frames:
        pil = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE) for f in best_frames]
        durs = [90] * len(pil)
        durs[-1] = 1500
        pil[0].save(record_gif_path, save_all=True, append_images=pil[1:], duration=durs, loop=0)
        print(f"Saved VLA rollout GIF -> {record_gif_path} ({len(pil)} frames)")

    return {
        "success_rate": float(np.mean(succ)),
        "mean_dist": float(np.mean(dists)),
        "mean_steps": float(np.mean(steps_l)),
        "frames": best_frames,
    }


def save_3panel_gif(
    panels: list[tuple[str, list[np.ndarray]]],
    out_path: str,
) -> None:
    if not all(len(f) > 0 for _, f in panels):
        return
    n = max(len(f) for _, f in panels)
    pil = []
    for i in range(n):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.1), dpi=90)
        for ax, (title, frs) in zip(axes, panels):
            ax.imshow(frs[min(i, len(frs) - 1)])
            ax.set_title(title, fontsize=9.5, fontweight="bold")
            ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        pil.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)
    durs = [90] * len(pil)
    durs[-1] = 1500
    pil[0].save(out_path, save_all=True, append_images=pil[1:], duration=durs, loop=0)
    print(f"Saved 3-panel comparison GIF -> {out_path} ({len(pil)} frames)")


def patch_all() -> None:
    q3.PixelShuffleProjector.forward = pixel_shuffle_forward
    q3.build_prefix_causal_mask = build_prefix_causal_mask
    q3.cosine_beta_schedule = cosine_beta_schedule
    q3.evaluate_chunk_mpc = evaluate_chunk_mpc


def main() -> None:
    ap = argparse.ArgumentParser(description="CW2 Q3 Track A VLA reference solutions")
    ap.add_argument("--demos", type=int, default=12)
    ap.add_argument("--vlm-epochs", type=int, default=12)
    ap.add_argument("--diff-epochs", type=int, default=180)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    patch_all()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f"CW2 Q3 — Track A VLA (nanoVLM + Diffusion Chunk-MPC) on PandaOmron [{device}]")
    print("=" * 78)

    ds = q3.collect_vla_dataset(n_episodes=args.demos, seed=0, img_size=64)
    eps = ds["episodes"]
    print(f"Collected {len(eps)} VLA episodes (total frames={sum(len(e['actions']) for e in eps)})")

    # 1. Train nanoVLM from scratch
    vlm_res = train_nanovlm(eps, epochs=args.vlm_epochs, seed=0, device=device)
    print(
        f"  nanoVLM ({args.vlm_epochs} ep) | Train CE: {vlm_res['train_ce'][-1]:.4f} | "
        f"Val CE: {vlm_res['val_ce'][-1]:.4f} | VQA Acc: {vlm_res['vqa_acc']*100:.1f}% "
        f"(Majority baseline: {vlm_res['maj_acc']*100:.1f}%)"
    )

    # 2. Build 4-step action chunks & train Diffusion Heads
    imgs, proprios, chunks = q3.build_action_chunks(eps, chunk_len=4)
    diff_under = train_diffusion_policy(imgs, proprios, chunks, vlm=None, epochs=30, seed=0, device=device)
    diff_prop = train_diffusion_policy(imgs, proprios, chunks, vlm=None, epochs=args.diff_epochs, seed=0, device=device)
    diff_vlm = train_diffusion_policy(imgs, proprios, chunks, vlm=vlm_res["model"], epochs=args.diff_epochs, seed=0, device=device)

    def _make_policy_fn(head, vlm_mod=None):
        def _fn(obs):
            p = mpe.extract_proprio(obs)
            p_t = torch.as_tensor(p, dtype=torch.float32, device=device).unsqueeze(0)
            if vlm_mod is not None:
                im = np.flipud(obs["agentview_image"]).copy()
                if im.shape[0] != 64:
                    im = np.asarray(Image.fromarray(im).resize((64, 64), Image.BILINEAR)).copy()
                im_t = torch.as_tensor(im, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
                with torch.no_grad():
                    emb = vlm_mod.encode_image(im_t).mean(dim=1)
                cond = torch.cat([p_t, emb], dim=-1)
            else:
                cond = p_t
            act_6 = head.sample(cond).squeeze(0).cpu().numpy().reshape(4, 6)
            return unpack_active(act_6).reshape(-1)
        return _fn

    ev_under = q3.evaluate_chunk_mpc(
        _make_policy_fn(diff_under["head"]),
        n_episodes=args.eval_episodes,
        exec_steps=1,
        use_camera=False,
        record_gif_path=os.path.join(args.out_dir, "q3_vla_proprio_30ep.gif"),
    )
    ev_prop = q3.evaluate_chunk_mpc(
        _make_policy_fn(diff_prop["head"]),
        n_episodes=args.eval_episodes,
        exec_steps=1,
        use_camera=False,
        record_gif_path=os.path.join(args.out_dir, "q3_vla_proprio_160ep.gif"),
    )
    ev_vlm_k1 = q3.evaluate_chunk_mpc(
        _make_policy_fn(diff_vlm["head"], vlm_res["model"]),
        n_episodes=args.eval_episodes,
        exec_steps=1,
        record_gif_path=os.path.join(args.out_dir, "q3_vla_vlm_k1.gif"),
    )
    ev_vlm_k4 = q3.evaluate_chunk_mpc(
        _make_policy_fn(diff_vlm["head"], vlm_res["model"]),
        n_episodes=args.eval_episodes,
        exec_steps=4,
        record_gif_path=os.path.join(args.out_dir, "q3_vla_vlm_k4.gif"),
    )
    save_3panel_gif(
        [
            ("Proprio (30 ep, Undertrained)", ev_under["frames"]),
            (f"Proprio-Only ({args.diff_epochs} ep)", ev_prop["frames"]),
            (f"VLM + Diffusion ({args.diff_epochs} ep, k=1)", ev_vlm_k1["frames"]),
        ],
        os.path.join(args.out_dir, "q3_vla_comparison_3panel.gif"),
    )

    print(f"  Diffusion Proprio (30 ep, k=1)   | Succ: {ev_under['success_rate']*100:5.1f}% | Dist: {ev_under['mean_dist']:.3f}m")
    print(f"  Diffusion Proprio ({args.diff_epochs:3d} ep, k=1) | Succ: {ev_prop['success_rate']*100:5.1f}% | Dist: {ev_prop['mean_dist']:.3f}m")
    print(f"  Diffusion +VLM    ({args.diff_epochs:3d} ep, k=1) | Succ: {ev_vlm_k1['success_rate']*100:5.1f}% | Dist: {ev_vlm_k1['mean_dist']:.3f}m | Steps: {ev_vlm_k1['mean_steps']:.1f}")
    print(f"  Diffusion +VLM    ({args.diff_epochs:3d} ep, k=4) | Succ: {ev_vlm_k4['success_rate']*100:5.1f}% | Dist: {ev_vlm_k4['mean_dist']:.3f}m | Steps: {ev_vlm_k4['mean_steps']:.1f}")

    # Plot 1: nanoVLM & Diffusion training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax1.plot(vlm_res["train_ce"], label="Train CE", color="#2563EB", lw=1.8)
    ax1.plot(vlm_res["val_ce"], label="Val CE", color="#DC2626", lw=1.8, ls="--")
    ax1.set(xlabel="Epoch", ylabel="Cross-Entropy Loss", title="nanoVLM Training (MobilePickPlace)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(diff_prop["losses"], label="Proprio-Only DDPM", color="#F59E0B", lw=1.8)
    ax2.plot(diff_vlm["losses"], label="Proprio + VLM DDPM", color="#16A34A", lw=1.8)
    ax2.set(xlabel="Epoch", ylabel="Epsilon-Prediction MSE", title="Conditional Diffusion Action Head")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, "q3_vla_training.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p1}")

    # Plot 2: Ablation & Chunk-MPC sweep (Success Rate + Final Cube-to-Pad Distance)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    labels = ["Proprio\n(30 ep)", f"Proprio\n({args.diff_epochs} ep)", "VLM+Diff\n(k=1)", "VLM+Diff\n(k=4)"]
    succs = [
        ev_under["success_rate"] * 100.0,
        ev_prop["success_rate"] * 100.0,
        ev_vlm_k1["success_rate"] * 100.0,
        ev_vlm_k4["success_rate"] * 100.0,
    ]
    dists = [ev_under["mean_dist"], ev_prop["mean_dist"], ev_vlm_k1["mean_dist"], ev_vlm_k4["mean_dist"]]
    cols = ["#94A3B8", "#F59E0B", "#16A34A", "#22C55E"]
    ax1.bar(labels, succs, color=cols, edgecolor="#0F172A")
    ax1.set(ylabel="Closed-Loop Success Rate (%)", ylim=(0, 105), title="Success Rate (Undertrain / VLM / k)")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(labels, dists, color=cols, edgecolor="#0F172A")
    ax2.set(ylabel="Final Cube-to-Pad Dist (m)", title="Final Cube-to-Pad Distance (lower is better)")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(args.out_dir, "q3_chunk_mpc.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p2}")
    print("CW2_Q3_SOLUTIONS_DONE")


if __name__ == "__main__":
    main()
