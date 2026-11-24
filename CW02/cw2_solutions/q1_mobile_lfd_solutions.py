"""Instructor reference solutions for COMP0037 2026/27 Coursework 2 — Question 1.

Patches `train_bc` and `evaluate_bc_policy` in `assessment/CW02/cw2_starter/q1_mobile_lfd.py`,
evaluates Behavioral Cloning (BC) dataset-size scaling, Standard BC vs. Noise-Augmented BC,
and nominal vs. out-of-distribution cube perturbations on `MobilePickPlaceEnv` (`PandaOmron`),
and saves:
  - `outputs/q1_lfd_comparison.png`
  - `outputs/q1_mobile_pick_place.gif` (side-by-side FSM Expert vs. BC policy rollout)

Usage (from repository root):
    .venv_robot\\Scripts\\python assessment\\CW02\\cw2_solutions\\q1_mobile_lfd_solutions.py
"""
from __future__ import annotations

import argparse
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
import q1_mobile_lfd as q1  # noqa: E402


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int = 16, act_dim: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, act_dim), nn.Tanh(),
        )
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = (x - self.obs_mean) / (self.obs_std + 1e-6)
        return self.net(xn)


def train_bc(
    obs_data: np.ndarray,
    act_data: np.ndarray,
    epochs: int = 200,
    noise_std: float = 0.0,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BCPolicy(obs_data.shape[1], act_data.shape[1])
    model.obs_mean.copy_(torch.as_tensor(obs_data.mean(axis=0), dtype=torch.float32))
    model.obs_std.copy_(torch.as_tensor(obs_data.std(axis=0) + 1e-6, dtype=torch.float32))

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    x_all = torch.as_tensor(obs_data, dtype=torch.float32)
    y_all = torch.as_tensor(act_data, dtype=torch.float32)
    n = len(obs_data)
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        ep_loss = 0.0
        n_b = 0
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            xb = x_all[idx]
            if noise_std > 0.0:
                xb = xb + torch.randn_like(xb) * noise_std
            pred = model(xb)
            loss = F.mse_loss(pred, y_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_b += 1
        sched.step()
        losses.append(ep_loss / max(1, n_b))
    return {"model": model, "losses": np.asarray(losses)}


def evaluate_bc_policy(
    policy: nn.Module | None,
    n_episodes: int = 5,
    seed: int = 200,
    cube_xy_noise: float = 0.04,
    horizon: int = 240,
    record_first: bool = False,
) -> dict:
    env = mpe.make_mobile_pick_place_env(
        use_camera_obs=record_first,
        camera_size=192 if record_first else 64,
        horizon=horizon,
        cube_xy_noise=cube_xy_noise,
        seed=seed,
    )
    expert = q1.MobilePickPlaceExpert()
    succ, final_dists, steps_list = [], [], []
    frames = []
    for ep in range(n_episodes):
        obs = env.reset()
        expert.reset()
        if record_first and ep == 0:
            frames.append(np.flipud(obs["agentview_image"]).copy())
        done_succ = False
        t_end = horizon
        for t in range(1, horizon + 1):
            if policy is None:
                act = expert.act(obs)
            else:
                with torch.no_grad():
                    p = torch.as_tensor(mpe.extract_proprio(obs)).unsqueeze(0)
                    act = policy(p).squeeze(0).numpy()
            obs, _, done, _ = env.step(act)
            if record_first and ep == 0 and (t % 2 == 0 or env._check_success()):
                frames.append(np.flipud(obs["agentview_image"]).copy())
            if env._check_success():
                done_succ = True
                t_end = t
                break
            if done:
                t_end = t
                break
        succ.append(float(done_succ))
        final_dists.append(float(np.linalg.norm(obs["cube_pos"][:2] - mpe.TARGET_PAD_POS[:2])))
        steps_list.append(t_end)
    env.close()
    return {
        "success_rate": float(np.mean(succ)),
        "mean_dist": float(np.mean(final_dists)),
        "mean_steps": float(np.mean(steps_list)),
        "frames": frames,
    }


def save_side_by_side_gif(frames_left: list[np.ndarray], frames_right: list[np.ndarray], out_path: str) -> None:
    if not frames_left or not frames_right:
        return
    n = max(len(frames_left), len(frames_right))
    pil_frames = []
    for i in range(n):
        fl = frames_left[min(i, len(frames_left) - 1)]
        fr = frames_right[min(i, len(frames_right) - 1)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.1), dpi=90)
        ax1.imshow(fl)
        ax1.set_title("FSM Expert (PandaOmron)", fontsize=10, fontweight="bold")
        ax1.axis("off")
        ax2.imshow(fr)
        ax2.set_title("Behavioral Cloning Policy", fontsize=10, fontweight="bold")
        ax2.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        pil_frames.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)
    durations = [100] * len(pil_frames)
    durations[-1] = 1500
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:], duration=durations, loop=0)
    print(f"Saved rollout GIF -> {out_path} ({len(pil_frames)} frames)")


def patch_all() -> None:
    q1.train_bc = train_bc
    q1.evaluate_bc_policy = evaluate_bc_policy


def main() -> None:
    ap = argparse.ArgumentParser(description="CW2 Q1 Behavioral Cloning reference solutions")
    ap.add_argument("--max-demos", type=int, default=16)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    patch_all()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 78)
    print("CW2 Q1 — MobilePickPlaceEnv (PandaOmron) Behavioral Cloning (BC)")
    print("=" * 78)
    full_data = q1.collect_demonstrations(n_episodes=args.max_demos, seed=0)
    print(
        f"Collected {len(full_data['ep_lens'])}/{args.max_demos} expert demos "
        f"(rate={full_data['success_rate']*100:.0f}%, mean_len={np.mean(full_data['ep_lens']):.1f} steps)"
    )

    # 1. Dataset-size scaling for Standard BC
    sizes = [s for s in (4, 8, args.max_demos) if s <= len(full_data["ep_lens"])]
    bc_scores, bc_dists = [], []
    best_bc = None
    cum_steps = np.cumsum([0] + full_data["ep_lens"])
    for n_ep in sizes:
        end_idx = int(cum_steps[n_ep])
        res = q1.train_bc(full_data["obs"][:end_idx], full_data["actions"][:end_idx], epochs=args.bc_epochs, noise_std=0.0, seed=0)
        ev = evaluate_bc_policy(res["model"], n_episodes=args.eval_episodes, seed=200)
        bc_scores.append(ev["success_rate"] * 100.0)
        bc_dists.append(ev["mean_dist"])
        best_bc = res
        print(f"  BC (N={n_ep:2d} demos)        | Succ: {ev['success_rate']*100:5.1f}% | Cube->Pad: {ev['mean_dist']:.3f}m | Steps: {ev['mean_steps']:.1f}")

    # 2. Noise-Augmented BC on full dataset (covariate-shift mitigation)
    noisy_bc = q1.train_bc(full_data["obs"], full_data["actions"], epochs=args.bc_epochs, noise_std=0.01, seed=0)
    noisy_ev = evaluate_bc_policy(noisy_bc["model"], n_episodes=args.eval_episodes, seed=200)
    print(f"  BC+Noise (N={len(full_data['ep_lens']):2d}, s=0.01) | Succ: {noisy_ev['success_rate']*100:5.1f}% | Cube->Pad: {noisy_ev['mean_dist']:.3f}m | Steps: {noisy_ev['mean_steps']:.1f}")

    # 3. Out-of-distribution cube perturbation test (0.08m vs nominal 0.04m)
    bc_pert = evaluate_bc_policy(best_bc["model"], n_episodes=args.eval_episodes, seed=300, cube_xy_noise=0.08)
    noisy_pert = evaluate_bc_policy(noisy_bc["model"], n_episodes=args.eval_episodes, seed=300, cube_xy_noise=0.08)
    exp_ev = evaluate_bc_policy(None, n_episodes=args.eval_episodes, seed=200, record_first=True)
    gif_model = noisy_bc["model"] if noisy_ev["success_rate"] >= bc_scores[-1] / 100.0 else best_bc["model"]
    bc_gif_ev = evaluate_bc_policy(gif_model, n_episodes=1, seed=200, record_first=True)
    print(f"  Perturbed (0.08m)        | BC: {bc_pert['success_rate']*100:5.1f}% | BC+Noise: {noisy_pert['success_rate']*100:5.1f}%")

    # Diagnostic 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    axes[0].plot(best_bc["losses"], color="#2563EB", lw=1.8, label="Standard BC")
    axes[0].plot(noisy_bc["losses"], color="#7C3AED", lw=1.8, ls="--", label="Noise-Aug BC")
    axes[0].set(xlabel="Epoch", ylabel="MSE Loss", title="BC Training Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].bar([f"N={s}" for s in sizes], bc_dists, color=["#93C5FD", "#3B82F6", "#1D4ED8"], edgecolor="#0F172A")
    axes[1].set(ylabel="Final Cube-to-Pad Dist (m)", title="BC Dataset-Size Scaling")
    axes[1].grid(axis="y", alpha=0.3)

    methods = ["Expert", "BC\n(nom)", "BC+Noise\n(nom)", "BC\n(pert)", "BC+Noise\n(pert)"]
    vals = [
        exp_ev["success_rate"] * 100.0,
        bc_scores[-1],
        noisy_ev["success_rate"] * 100.0,
        bc_pert["success_rate"] * 100.0,
        noisy_pert["success_rate"] * 100.0,
    ]
    cols = ["#16A34A", "#2563EB", "#7C3AED", "#60A5FA", "#A78BFA"]
    axes[2].bar(methods, vals, color=cols, edgecolor="#0F172A")
    axes[2].set(ylabel="Success Rate (%)", ylim=(0, 105), title="Nominal vs. Perturbed Eval")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    p_out = os.path.join(args.out_dir, "q1_lfd_comparison.png")
    fig.savefig(p_out, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p_out}")

    gif_out = os.path.join(args.out_dir, "q1_mobile_pick_place.gif")
    save_side_by_side_gif(exp_ev["frames"], bc_gif_ev["frames"], gif_out)
    print("CW2_Q1_SOLUTIONS_DONE")


if __name__ == "__main__":
    main()
