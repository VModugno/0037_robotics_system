"""Week 08 - VLA lab, part 3: attach a diffusion action head (pi0-style).

A VLA is a VLM plus an action expert. The final stage of this lab attaches
a DDPM action head to the VLM you just trained -- the same architecture
move pi0 (Physical Intelligence) and TRI's Foundry-VLA make:

    image --TinyViT--> --projector--> VLM embedding ┐
                                                    ├──(cond)──> diffusion
    proprio (12-D) ---------------------------------┘              head -> action
                                                                     chunk (H x 7)

The head denoises an ACTION CHUNK (H=4 consecutive 7-D OSC_POSE actions)
conditioned on perception -- DDPM, epsilon-prediction, cosine schedule,
ancestral sampling. Small enough to train on CPU in minutes on the data
from vlm_data.py.

Without --vlm-checkpoint the head conditions on proprio only (an honest
MLP-BC-equivalent baseline); with it, the VLM's visual embedding joins the
conditioning -- the full VLA wiring, trained end-to-end on ~600 frames.

TODO map:
    T5  cosine_beta_schedule   the DDPM noise schedule
    T6  q_sample               forward noising
    T7  p_losses               epsilon-prediction MSE
    T8  sample                 ancestral sampling loop
    T9  build_windows          (state, action-chunk) windows from episodes
    T10 evaluate_policy        chunk-MPC loop in the live env

Usage:
    pixi run --manifest-path ../pixi.toml -e week08 python diffusion_action_head.py \
        --data vlm_data --epochs 30 [--vlm-checkpoint nanovlm.pt] --eval-episodes 10
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from vlm_data import make_env, state_features  # noqa: E402  (same env + 12-D features as part 1)

HORIZON = 4          # action chunk length (pi0 predicts chunks, not steps)
ACT_DIM = 7
T_DIFF = 100          # DDPM timesteps
COND_DIM = 256
TIME_DIM = 64


# --------------------------------------------------------------------------- #
# the denoiser: FiLM-MLP conditioned on (time, perception)
# --------------------------------------------------------------------------- #


def timestep_embedding(t, dim=TIME_DIM):
    """Sinusoidal time embedding (complete)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return torch.nn.functional.gelu(self.lin(x) * (1 + scale) + shift)


class ActionDenoiser(nn.Module):
    """epsilon(x_t, cond, t) -- predicts the noise in a chunk (complete)."""

    def __init__(self, act_dim=ACT_DIM, horizon=HORIZON, cond_dim=COND_DIM):
        super().__init__()
        self.in_dim = act_dim * horizon
        self.cond_net = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU())
        self.time_net = nn.Sequential(nn.Linear(TIME_DIM, cond_dim), nn.SiLU())
        self.in_proj = nn.Linear(self.in_dim, 256)
        self.blocks = nn.ModuleList([FiLMBlock(256, cond_dim) for _ in range(3)])
        self.out = nn.Linear(256, self.in_dim)

    def forward(self, x_t, cond, t):
        """x_t (B, in_dim), cond (B, cond_dim), t (B,) -> epsilon (B, in_dim)."""
        c = self.cond_net(cond) + self.time_net(timestep_embedding(t))
        h = self.in_proj(x_t)
        for blk in self.blocks:
            h = h + blk(h, c)
        return self.out(h)


# --------------------------------------------------------------------------- #
# TODO T5-T8: the DDPM itself
# --------------------------------------------------------------------------- #


def cosine_beta_schedule(T=T_DIFF, s=0.008):
    """TODO T5. Return the betas (T,) from the cosine alpha-bar schedule.

    alpha_bar(t) = cos^2( ((t/T + s) / (1 + s)) * pi/2 ) / cos^2( s*pi/2 )
    for t = 0..T, then beta_t = 1 - alpha_bar(t) / alpha_bar(t-1)
    (clip betas into (0, 0.999]). This is Nichol & Dhariwal's "improved
    DDPM" schedule -- flat in the middle, gentle at both ends.
    """
    raise NotImplementedError("T5: cosine beta schedule (see docstring)")


class DiffusionHead:
    """The DDPM over flattened action chunks.

    The schedule buffers are attached by set_schedule() once T5 provides
    the betas (kept explicit so the student TODO owns the schedule).
    """

    def __init__(self, denoiser, device, T=T_DIFF):
        self.net = denoiser
        self.device = device
        self.T = T
        self.betas = None

    def set_schedule(self, betas):
        """Store the schedule tensors (complete -- T5 feeds this)."""
        betas = betas.to(self.device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1macp = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise):
        """TODO T6. Forward process: x_t = sqrt(a_bar_t) x0 + sqrt(1 - a_bar_t) eps.

        x0 (B, D) clean chunks, t (B,) ints in [0, T), noise (B, D) ~ N(0,1).
        Gather sqrt_acp / sqrt_1macp at t (reshape to (B, 1) to broadcast).
        """
        raise NotImplementedError("T6: forward noising (see docstring)")

    def p_losses(self, x0, cond, t, noise):
        """TODO T7. epsilon-prediction loss.

        Noise x0 into x_t (T6), ask the denoiser to recover `noise` from
        (x_t, cond, t), return the MSE between prediction and `noise`.
        """
        raise NotImplementedError("T7: epsilon loss (see docstring)")

    @torch.no_grad()
    def sample(self, cond, n_samples=None):
        """TODO T8. Ancestral sampling: T-1 -> 0.

        Start from x ~ N(0, I) of shape (B, D) where B = cond.size(0).
        For t = T-1, T-2, ..., 0 (a torch scalar tensor per step):
            eps_hat = net(x, cond, t)
            x = (1/sqrt(alpha_t)) * (x - (beta_t / sqrt(1-a_bar_t)) * eps_hat)
            if t > 0: x += sqrt(beta_t) * z,  z ~ N(0, I)
        Return the final x (B, D). Batched over the schedule arrays --
        precompute beta/alpha tensors in set_schedule and index them with
        a python int (they are already on the right device).
        """
        raise NotImplementedError("T8: ancestral sampling (see docstring)")


# --------------------------------------------------------------------------- #
# data windows (T9)
# --------------------------------------------------------------------------- #


def load_episodes(data_dir):
    """[(states (T,12), actions (T,7), frames (T,3,64,64) | None)] (complete)."""
    episodes = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        npz = np.load(os.path.join(data_dir, fname))
        has_frames = "frames" in npz
        episodes.append(
            (
                npz["states"].astype(np.float32),
                npz["actions"].astype(np.float32),
                npz["frames"] if has_frames else None,
            )
        )
    return episodes


def build_windows(episodes, horizon=HORIZON):
    """TODO T9. Windows for imitation: (state_t, action_chunk, episode_idx, t).

    For every episode (states, actions, _) and every t in
    range(len(actions) - horizon + 1):
        state  = states[t]                    (12,)
        chunk  = actions[t : t + horizon]      (horizon, 7), flattened to (D,)
    Return (states (N,12), chunks (N, horizon*7), ep_idx (N,), t (N,)) as
    numpy arrays. Only WITHIN-episode windows -- never cross the episode
    boundary (the reset teleport would teach the head garbage).
    """
    raise NotImplementedError("T9: windowed dataset (see docstring)")


# --------------------------------------------------------------------------- #
# VLM conditioning glue (complete)
# --------------------------------------------------------------------------- #


class VLMEmbedder:
    """Mean-pooled vision tokens from the NanoVLM of part 2 (256-D)."""

    def __init__(self, ckpt_path, device):
        import nanovlm_from_scratch as nvl

        # weights_only=False: our own checkpoint (nanovlm.pt) stores the
        # vocab list; torch>=2.6 defaults to weights_only=True and refuses it
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        tok_len = len(ckpt["vocab"])
        self.model = nvl.NanoVLM(tok_len).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def embed_frames(self, frames_u8):
        """(B, H, W, 3) uint8 -> (B, 256) float embeddings."""
        x = torch.from_numpy(frames_u8).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return self.model.vision_tokens(x).mean(dim=1).cpu().numpy()


# --------------------------------------------------------------------------- #
# training + evaluation driver (complete apart from the TODOs)
# --------------------------------------------------------------------------- #


def train(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    episodes = load_episodes(args.data)
    try:
        states, chunks, ep_idx, ts = build_windows(episodes, HORIZON)
    except NotImplementedError as e:
        print(f"[!] {e}\n    Complete TODO T9 in diffusion_action_head.py, then re-run.")
        return None
    print(f"windows: {len(chunks)} chunks of H={HORIZON} from {len(episodes)} episodes")

    mu, sd = states.mean(0), states.std(0) + 1e-8
    states_n = (states - mu) / sd

    embedder = None
    frame_emb = None
    if args.vlm_checkpoint:
        embedder = VLMEmbedder(args.vlm_checkpoint, device)
        print("conditioning: VLM embedding (256-D) + proprio (12-D)")
        frame_emb = {}
        for ei, (_, _, frames) in enumerate(episodes):
            if frames is not None:
                frame_emb[ei] = embedder.embed_frames(frames)
    else:
        print("conditioning: proprio only (12-D) -- pass --vlm-checkpoint for the full VLA)")

    def cond_matrix(ei_arr, t_arr):
        """(N, cond_dim) conditioning for the window rows (complete)."""
        prop = states_n  # proprio rows already aligned with the windows
        if frame_emb is None:
            return prop
        embs = np.stack([frame_emb[ei][t] for ei, t in zip(ei_arr, t_arr)])
        return np.concatenate([prop, embs], axis=1)

    conds = cond_matrix(ep_idx, ts)
    cond_dim = conds.shape[1]

    denoiser = ActionDenoiser(cond_dim=cond_dim).to(device)
    ddpm = DiffusionHead(denoiser, device)
    try:
        betas = cosine_beta_schedule(T_DIFF)
    except NotImplementedError as e:
        print(f"[!] {e}\n    Complete TODO T5 in diffusion_action_head.py, then re-run.")
        return None
    ddpm.set_schedule(betas)

    X = torch.from_numpy(chunks).float().to(device)
    C = torch.from_numpy(conds).float().to(device)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(X))
    n_val = max(1, int(0.1 * len(X)))
    val_idx, train_idx = order[:n_val], order[n_val:]

    optim = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    history = []
    for epoch in range(1, args.epochs + 1):
        tot, n = 0.0, 0
        for i in range(0, len(train_idx), args.batch_size):
            idx = train_idx[i : i + args.batch_size]
            x0 = X[idx]
            cond = C[idx]
            t = torch.from_numpy(rng.integers(0, T_DIFF, size=len(idx))).to(device)
            noise = torch.randn_like(x0)
            try:
                loss = ddpm.p_losses(x0, cond, t, noise)
            except NotImplementedError as e:
                print(f"[!] {e}\n    Complete TODO T6/T7, then re-run.")
                return None
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        with torch.no_grad():
            idx = val_idx
            vt = 0.0
            for i in range(0, len(idx), args.batch_size):
                b = idx[i : i + args.batch_size]
                t = torch.from_numpy(rng.integers(0, T_DIFF, size=len(b))).to(device)
                noise = torch.randn_like(X[b])
                vt += ddpm.p_losses(X[b], C[b], t, noise).item() * len(b)
            vt /= max(len(idx), 1)
        history.append((tot / max(n, 1), vt))
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}: train {tot / max(n, 1):.5f}  val {vt:.5f}")

    torch.save(
        {
            "denoiser": denoiser.state_dict(),
            "cond_dim": cond_dim,
            "horizon": HORIZON,
            "state_mu": mu,
            "state_sd": sd,
            "uses_vlm": frame_emb is not None,
        },
        args.out,
    )
    print(f"saved -> {args.out}")

    _, ax = plt.subplots(figsize=(6, 4))
    ax.plot([h[0] for h in history], label="train")
    ax.plot([h[1] for h in history], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("epsilon-MSE")
    ax.legend()
    ax.set_title("diffusion action head")
    ax.figure.savefig(args.plot, dpi=120, bbox_inches="tight")
    print(f"plot -> {args.plot}")

    return ddpm, embedder, mu, sd, frame_emb is not None


def evaluate_policy(ddpm, embedder, mu, sd, uses_vlm, args):
    """TODO T10. Chunk-MPC evaluation in the live Lift env.

    Per control step (up to args.horizon steps, unless success):
        1. state = vlm_data.state_features(obs)  (12-D) -> normalize with
           (mu, sd) -> torch (1, 12) on ddpm.device;
        2. cond = state tensor; if uses_vlm, embed the current agentview
           frame (uint8 HxWx3 -> VLMEmbedder.embed_frames) and concatenate;
        3. chunk = ddpm.sample(cond).view(HORIZON, ACT_DIM) on CPU numpy;
        4. execute the FIRST action of the chunk (re-plan every step =
           MPC), then env.step(action);
        5. stop early when env._check_success().
    Per episode print (episode, steps, success); at the end print the
    summary "eval: X/Y successes, mean steps Z" and RETURN the success
    count. Episodes use np.random.seed(args.seed + 100 + ep) for placement.
    """
    raise NotImplementedError("T10: evaluation loop (see docstring)")


def main():
    parser = argparse.ArgumentParser(description="Week 08 diffusion action head")
    parser.add_argument("--data", type=str, default="vlm_data")
    parser.add_argument("--epochs", type=int, default=150)  # 150 verified: 30 epochs is undertrained (eval 0/10); see LAB.md E2
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vlm-checkpoint", type=str, default="")
    parser.add_argument("--out", type=str, default="diffusion_head.pt")
    parser.add_argument("--plot", type=str, default="diffusion_training.png")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=300, help="env steps per eval episode")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    trained = train(args)
    if trained is None:
        return
    if not args.skip_eval:
        try:
            evaluate_policy(*trained, args)
        except NotImplementedError as e:
            print(f"[!] {e}\n    Complete TODO T10, then re-run.")


if __name__ == "__main__":
    main()
