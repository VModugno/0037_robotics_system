"""Instructor solutions for week 08 (VLA lab): reference T1-T10 + stage runner.

Run from labs/week08_vla (outputs land in the cwd, like the lab scripts):

    pixi run --manifest-path ../pixi.toml -e week08 python \
        ../solutions/week08_vla_solutions.py --stage data
    ... then --stage vlm, --stage diffusion, --stage diffvlm
    or --stage dryrun for an instant patch self-check.

Stages set sys.argv and call the lab mains after monkey-patching the TODOs,
so the printed numbers are exactly what a student with correct TODOs gets.
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAB_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "week08_vla")
)
sys.path.insert(0, LAB_DIR)

import diffusion_action_head as dah  # noqa: E402
import nanovlm_from_scratch as nvl  # noqa: E402
import vlm_data  # noqa: E402

PAD_ID = 0   # Tokenizer itos order: <pad>, <bos>, <eos>
EOS_ID = 2


# --------------------------------------------------------------------------- #
# reference implementations for the student TODOs
# --------------------------------------------------------------------------- #


def _tinyvit_init(self, img=nvl.IMG_SIZE, patch=nvl.PATCH, dim=nvl.VIT_DIM,
                  layers=nvl.VIT_LAYERS, heads=nvl.VIT_HEADS):
    """Patched __init__: also creates the patch conv that T1 needs."""
    nn.Module.__init__(self)
    self.grid = img // patch
    self.patch = patch
    self.dim = dim
    self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
    self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
    self.blocks = nn.ModuleList([nvl.ViTBlock(dim, heads) for _ in range(layers)])
    self.norm = nn.LayerNorm(dim)
    nn.init.trunc_normal_(self.pos, std=0.02)


def _patch_embed(self, x):
    """T1: conv -> flatten row-major -> +pos -> blocks -> norm."""
    x = self.proj(x)                              # (B, dim, gh, gh)
    x = torch.flatten(x, 2).transpose(1, 2)       # (B, gh*gh, dim)
    x = x + self.pos
    for blk in self.blocks:
        x = blk(x)
    return self.norm(x)


def _projector_forward(self, patch_tokens):
    """T2: pixel shuffle then the single linear (nanoVLM ModalityProjector)."""
    b, t, c = patch_tokens.shape
    grid = int(math.sqrt(t))
    x = nvl.pixel_shuffle_tokens(patch_tokens, grid, self.factor, c)
    return self.proj(x)


def _prefix_mask(P, L, device):
    """(P+L, P+L) float additive mask, 0.0=visible and -inf=blocked.

    Bool masks are ambiguous on torch >= 2.x: MultiheadAttention reads
    them differently in train vs eval paths (a bool mask that trains
    fine NaNs at eval). A 0/-inf float mask is unambiguous everywhere.
    """
    S = P + L
    mask = torch.zeros(S, S, device=device)
    blocked = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
    )
    mask[P:, P:] = blocked.float().masked_fill(blocked, float("-inf"))
    return mask


def _forward_loss(self, images, tokens):
    """T3: prefix concat + prefix-aware mask + CE on text positions only."""
    img = self.vision_tokens(images)               # (B, P, D)
    P = img.size(1)
    x = torch.cat([img, self.lm.embed(tokens)], dim=1)   # (B, P+L, D)
    mask = _prefix_mask(P, tokens.size(1), x.device)
    for blk in self.lm.blocks:
        x = blk(x, mask)
    logits = self.lm.head(self.lm.norm(x))         # (B, P+L, V)
    logits_text = logits[:, P:, :]                 # (B, L, V)
    targets = torch.full_like(tokens, PAD_ID)
    targets[:, :-1] = tokens[:, 1:]
    # ignore padding targets: the model learns <eos>, not <pad><pad><pad>
    loss = F.cross_entropy(
        logits_text.reshape(-1, logits_text.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_ID,
    )
    n_text = int((targets != PAD_ID).sum().item())
    return loss, n_text


@torch.no_grad()
def _generate(self, images, prompt_tokens, max_new=nvl.MAX_TEXT):
    """T4: greedy decoding, recompute the prefix every step (v0.1 style)."""
    img = self.vision_tokens(images)
    P = img.size(1)
    B = prompt_tokens.size(0)
    out = torch.zeros(B, max_new, dtype=torch.long, device=prompt_tokens.device)
    seq = prompt_tokens
    for step in range(max_new):
        x = torch.cat([img, self.lm.embed(seq)], dim=1)
        mask = _prefix_mask(P, seq.size(1), x.device)
        for blk in self.lm.blocks:
            x = blk(x, mask)
        logits = self.lm.head(self.lm.norm(x))
        nxt = logits[:, -1].argmax(dim=-1)
        out[:, step] = nxt
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        if bool((nxt == EOS_ID).all()):
            break
    for b in range(B):                             # trim each row at its <eos>
        hits = (out[b] == EOS_ID).nonzero().flatten()
        if len(hits):
            out[b, hits[0].item() + 1:] = PAD_ID
    return out


def _cosine_beta_schedule(T=dah.T_DIFF, s=0.008):
    """T5: Nichol & Dhariwal cosine alpha-bar schedule."""
    t = torch.arange(T + 1, dtype=torch.float32)
    abar = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    abar = abar / abar[0]
    betas = 1.0 - abar[1:] / abar[:-1]
    return betas.clamp(1e-6, 0.999)


def _q_sample(self, x0, t, noise):
    """T6: x_t = sqrt(a_bar_t) x0 + sqrt(1 - a_bar_t) eps."""
    sa = self.sqrt_acp[t][:, None]
    s1a = self.sqrt_1macp[t][:, None]
    return sa * x0 + s1a * noise


def _p_losses(self, x0, cond, t, noise):
    """T7: epsilon-prediction MSE."""
    x_t = self.q_sample(x0, t, noise)
    eps = self.net(x_t, cond, t)
    return F.mse_loss(eps, noise)


@torch.no_grad()
def _sample(self, cond, n_samples=None):
    """T8: ancestral sampling from N(0, I) down to t=0."""
    B = cond.size(0)
    x = torch.randn(B, self.net.in_dim, device=self.device)
    for t in range(self.T - 1, -1, -1):
        tt = torch.full((B,), t, dtype=torch.long, device=self.device)
        eps = self.net(x, cond, tt)
        beta_t = self.betas[t]
        abar_t = self.alphas_cumprod[t]
        x = (x - (beta_t / torch.sqrt(1.0 - abar_t)) * eps) / torch.sqrt(1.0 - beta_t)
        if t > 0:
            x = x + torch.sqrt(beta_t) * torch.randn_like(x)
    return x


def _build_windows(episodes, horizon=dah.HORIZON):
    """T9: within-episode (state, chunk) windows."""
    states, chunks, eidx, tidx = [], [], [], []
    for ei, (ep_states, ep_actions, _) in enumerate(episodes):
        for t in range(len(ep_actions) - horizon + 1):
            states.append(ep_states[t])
            chunks.append(ep_actions[t : t + horizon].reshape(-1))
            eidx.append(ei)
            tidx.append(t)
    return (
        np.stack(states).astype(np.float32),
        np.stack(chunks).astype(np.float32),
        np.asarray(eidx),
        np.asarray(tidx),
    )


def _evaluate_policy(ddpm, embedder, mu, sd, uses_vlm, args):
    """T10: chunk-MPC in the live Lift env (execute first action, replan)."""
    env = dah.make_env(image_size=64, horizon=args.horizon)
    succ, steps_hist = [], []
    for ep in range(args.eval_episodes):
        np.random.seed(args.seed + 100 + ep)
        obs = env.reset()
        success = False
        steps = 0
        for steps in range(1, args.horizon + 1):
            state = dah.state_features(obs)
            cond = torch.from_numpy(((state - mu) / sd)).float().unsqueeze(0).to(ddpm.device)
            if uses_vlm:
                frame = np.asarray(obs["agentview_image"], dtype=np.uint8)[None]
                emb = embedder.embed_frames(frame)
                cond = torch.cat([cond, torch.from_numpy(emb).float().to(ddpm.device)], dim=1)
            chunk = ddpm.sample(cond).view(dah.HORIZON, dah.ACT_DIM).cpu().numpy()
            obs, *_ = env.step(chunk[0])
            if env._check_success():
                success = True
                break
        succ.append(success)
        steps_hist.append(steps)
        print(f"episode {ep:2d}: {steps:3d} steps, success={success}")
    n = sum(succ)
    print(f"eval: {n}/{args.eval_episodes} successes, mean steps {np.mean(steps_hist):.1f}")
    env.close()
    return n


def patch_all():
    nvl.TinyViT.__init__ = _tinyvit_init
    nvl.TinyViT.patch_embed = _patch_embed
    nvl.Projector.forward = _projector_forward
    nvl.NanoVLM.forward_loss = _forward_loss
    nvl.NanoVLM.generate = _generate
    dah.cosine_beta_schedule = _cosine_beta_schedule
    dah.DiffusionHead.q_sample = _q_sample
    dah.DiffusionHead.p_losses = _p_losses
    dah.DiffusionHead.sample = _sample
    dah.build_windows = _build_windows
    dah.evaluate_policy = _evaluate_policy


# --------------------------------------------------------------------------- #
# dryrun: instant check that every patch target is callable and sane
# --------------------------------------------------------------------------- #


def dryrun():
    torch.manual_seed(0)
    vit = nvl.TinyViT()
    x = torch.randn(2, 3, nvl.IMG_SIZE, nvl.IMG_SIZE)
    toks = vit(x)
    assert toks.shape == (2, 64, nvl.VIT_DIM), toks.shape

    model = nvl.NanoVLM(32)
    loss, n_text = model.forward_loss(x, torch.randint(3, 32, (2, 6)))
    assert loss.requires_grad and n_text > 0
    out = model.generate(x, torch.randint(3, 32, (2, 4)), max_new=5)
    assert out.shape == (2, 5)

    betas = dah.cosine_beta_schedule()
    assert betas.shape == (dah.T_DIFF,) and float(betas.min()) > 0
    denoiser = dah.ActionDenoiser(cond_dim=12)
    ddpm = dah.DiffusionHead(denoiser, torch.device("cpu"))
    ddpm.set_schedule(betas)
    x0 = torch.randn(3, dah.HORIZON * dah.ACT_DIM)
    cond = torch.randn(3, 12)
    t = torch.tensor([5, 50, 95])
    ddpm.p_losses(x0, cond, t, torch.randn_like(x0))
    s = ddpm.sample(cond)
    assert s.shape == (3, dah.HORIZON * dah.ACT_DIM)

    eps = [
        (np.arange(12 * 8, dtype=np.float32).reshape(8, 12),
         np.arange(7 * 8, dtype=np.float32).reshape(8, 7), None),
    ]
    w = dah.build_windows(eps)
    assert w[0].shape == (5, 12) and w[1].shape == (5, 28)
    print("PATCHES OK (T1-T10 targets exercised)")


# --------------------------------------------------------------------------- #
# stage runner
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="week08 instructor solutions")
    parser.add_argument("--stage", required=True,
                        choices=["dryrun", "data", "vlm", "diffusion", "diffvlm", "all"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--data", type=str, default="vlm_data")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15, help="nanoVLM epochs")
    parser.add_argument("--diff-epochs", type=int, default=150, help="diffusion epochs (150 verified; 30 undertrained)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--vlm-out", type=str, default="nanovlm.pt")
    parser.add_argument("--vlm-plot", type=str, default="nanovlm_training.png")
    parser.add_argument("--diff-out", type=str, default="diffusion_head.pt")
    parser.add_argument("--diffvlm-out", type=str, default="diffusion_head_vlm.pt")
    parser.add_argument("--diff-plot", type=str, default="diffusion_training.png")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    patch_all()

    if args.stage == "dryrun":
        dryrun()
        return

    stages = {
        "data": ["vlm_data.py", "--episodes", str(args.episodes),
                 "--directory", args.data, "--image-size", str(args.image_size),
                 "--horizon", str(args.horizon), "--seed", str(args.seed)],
        "vlm": ["nanovlm_from_scratch.py", "--data", args.data,
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--seed", str(args.seed), "--out", args.vlm_out,
                "--plot", args.vlm_plot] + (["--skip-eval"] if args.skip_eval else []),
        "diffusion": ["diffusion_action_head.py", "--data", args.data,
                      "--epochs", str(args.diff_epochs), "--batch-size", str(args.batch_size),
                      "--seed", str(args.seed), "--out", args.diff_out,
                      "--plot", args.diff_plot,
                      "--eval-episodes", str(args.eval_episodes),
                      "--horizon", str(args.horizon)]
                     + (["--skip-eval"] if args.skip_eval else []),
        "diffvlm": ["diffusion_action_head.py", "--data", args.data,
                    "--epochs", str(args.diff_epochs), "--batch-size", str(args.batch_size),
                    "--seed", str(args.seed), "--vlm-checkpoint", args.vlm_out,
                    "--out", args.diffvlm_out, "--plot", "diffusion_vlm_training.png",
                    "--eval-episodes", str(args.eval_episodes),
                    "--horizon", str(args.horizon)]
                   + (["--skip-eval"] if args.skip_eval else []),
    }
    order = {
        "data": ["data"],
        "vlm": ["vlm"],
        "diffusion": ["diffusion"],
        "diffvlm": ["diffvlm"],
        "all": ["data", "vlm", "diffusion", "diffvlm"],
    }[args.stage]

    for stage in order:
        print(f"\n===== stage {stage} =====")
        sys.argv = stages[stage]
        if stage == "data":
            vlm_data.main()
        elif stage in ("vlm",):
            nvl.main()
        else:
            dah.main()


if __name__ == "__main__":
    main()
