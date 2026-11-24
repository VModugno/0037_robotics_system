"""Instructor reference solutions for COMP0037 2026/27 Coursework 1 — Question 3.

Covers both parts of Q3 on `assessment/cw1_starter/q3_sterile_arm.py`:
  - Q3.b/c: Standard DQN vs. Double DQN vs. No-Target-Net on SterileArmEnv(mode='discrete')
  - Q3.d/e: Patches TODO(CW1-Q3d) (`cbf_safety_shield`) with a dual-barrier
            (joint-limit + circular centrifuge obstacle) shield and compares
            Unshielded A2C, Soft-Penalty A2C, and Dual-Barrier CBF-Shielded A2C
            on SterileArmEnv(mode='continuous').

Usage (from repository root):
    .venv_rl\\Scripts\\python assessment\\cw1_solutions\\q3_sterile_arm_solutions.py
    .venv_rl\\Scripts\\python assessment\\cw1_solutions\\q3_sterile_arm_solutions.py --gif-seed 10
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
STARTER_DIR = os.path.normpath(os.path.join(CUR_DIR, os.pardir, "cw1_starter"))
OUT_DIR = os.path.join(CUR_DIR, "outputs")
sys.path.insert(0, STARTER_DIR)

import q3_sterile_arm as q3  # noqa: E402


# --------------------------------------------------------------------------- #
# Q3.b/c: DQN & Double DQN on SterileArmEnv(mode="discrete")
# --------------------------------------------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000, obs_dim: int = 6):
        self.cap = capacity
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.sn = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, s, a, r, sn, terminated):
        self.s[self.pos] = s
        self.a[self.pos] = a
        self.r[self.pos] = r
        self.sn[self.pos] = sn
        self.d[self.pos] = float(terminated)
        self.pos = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.sn[idx], self.d[idx]


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = 6, n_actions: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_dqn_variant(
    episodes: int,
    seed: int,
    double_dqn: bool = True,
    use_target_net: bool = True,
    device: str = "cpu",
) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = q3.SterileArmEnv(mode="discrete", seed=seed)
    q_net = QNetwork().to(device)
    t_net = QNetwork().to(device)
    t_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    buf = ReplayBuffer()

    gamma = 0.98
    batch_size = 64
    target_sync = 150
    step_count = 0
    returns = []

    for ep in range(episodes):
        s = env.reset()
        eps = max(0.05, 0.9 - 0.85 * (ep / max(1, int(0.75 * episodes))))
        ep_ret = 0.0
        done = False
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(0, 9))
            else:
                with torch.no_grad():
                    a = int(q_net(torch.as_tensor(s, device=device).unsqueeze(0)).argmax(1).item())
            sn, r, done, info = env.step(a)
            if info["obs_hit"]:
                r -= 0.5
            # Bootstrap on time-limit truncations; only true success terminates
            terminated = bool(info["success"])
            buf.push(s, a, r, sn, terminated)
            s = sn
            ep_ret += r
            step_count += 1

            if buf.size >= 256:
                bs, ba, br, bsn, bd = buf.sample(batch_size, rng)
                bs_t = torch.as_tensor(bs, device=device)
                ba_t = torch.as_tensor(ba, device=device).unsqueeze(1)
                br_t = torch.as_tensor(br, device=device)
                bsn_t = torch.as_tensor(bsn, device=device)
                bd_t = torch.as_tensor(bd, device=device)

                q_pred = q_net(bs_t).gather(1, ba_t).squeeze(1)
                eval_net = t_net if use_target_net else q_net
                with torch.no_grad():
                    if double_dqn and use_target_net:
                        a_star = q_net(bsn_t).argmax(1, keepdim=True)
                        next_q = t_net(bsn_t).gather(1, a_star).squeeze(1)
                    else:
                        next_q = eval_net(bsn_t).max(1).values
                    target = br_t + gamma * (1.0 - bd_t) * next_q

                loss = F.smooth_l1_loss(q_pred, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()

                if use_target_net and step_count % target_sync == 0:
                    t_net.load_state_dict(q_net.state_dict())
        returns.append(ep_ret)

    # Greedy evaluation over 20 episodes
    succ, obs_hits, final_dists, chatters = [], [], [], []
    for k in range(20):
        s = env.reset(seed=1000 + seed * 50 + k)
        done = False
        ep_obs = 0
        chatter = 0.0
        q_prev = env.q.copy()
        while not done:
            with torch.no_grad():
                a = int(q_net(torch.as_tensor(s, device=device).unsqueeze(0)).argmax(1).item())
            s, _, done, info = env.step(a)
            chatter += float(np.sum(np.abs(env.q - q_prev)))
            q_prev = env.q.copy()
            ep_obs += int(info["obs_hit"])
        succ.append(float(info["success"]))
        obs_hits.append(float(ep_obs > 0))
        final_dists.append(info["dist"])
        chatters.append(chatter)

    return {
        "returns": np.asarray(returns),
        "success_rate": float(np.mean(succ)),
        "obs_collision_rate": float(np.mean(obs_hits)),
        "final_dist": float(np.mean(final_dists)),
        "chattering": float(np.mean(chatters)),
        "model": q_net,
    }


# --------------------------------------------------------------------------- #
# Q3.d/e: Dual-Barrier CBF Shield & A2C on SterileArmEnv(mode="continuous")
# --------------------------------------------------------------------------- #

def cbf_safety_shield(q: np.ndarray, u_nom: np.ndarray, alpha_obs: float = 0.5) -> np.ndarray:
    """Reference dual-barrier CBF shield: joint limits + circular centrifuge obstacle."""
    u = np.clip(np.asarray(u_nom, dtype=np.float64).copy(), -q3.MAX_VEL, q3.MAX_VEL)
    r_safe = q3.OBS_RADIUS + 0.020  # Euler curvature margin
    u_lo = np.maximum(-q3.MAX_VEL, (q3.Q_MIN - q) / q3.DT)
    u_hi = np.minimum(q3.MAX_VEL, (q3.Q_MAX - q) / q3.DT)

    ee = q3.forward_kinematics(float(q[0]), float(q[1]))
    diff = ee - q3.OBS_CENTER
    h_obs = float(np.dot(diff, diff) - r_safe ** 2)
    J = q3.jacobian(float(q[0]), float(q[1]))
    g = 2.0 * (J.T @ diff) * q3.DT
    gn2 = float(np.dot(g, g)) + 1e-8

    activated = False
    for _ in range(3):
        margin = float(np.dot(g, u) + alpha_obs * h_obs)
        if margin < 0.0:
            activated = True
            u = u - (margin / gn2) * g
        u = np.clip(u, u_lo, u_hi)

    # Break antipodal local minimum with a small tangential slide around the centrifuge
    if activated and np.linalg.norm(u) < 0.25 * np.linalg.norm(u_nom):
        t_cart = np.array([-diff[1], diff[0]], dtype=np.float64) / (np.linalg.norm(diff) + 1e-8)
        t_q = J.T @ t_cart
        tn = float(np.linalg.norm(t_q))
        if tn > 1e-6:
            u = np.clip(u + 0.35 * (t_q / tn), u_lo, u_hi)

    # Discrete-step nonlinear curvature guard: strictly guarantee ||FK(q_next) - c|| >= OBS_RADIUS
    for _ in range(8):
        q_cand = np.clip(q + u * q3.DT, q3.Q_MIN, q3.Q_MAX)
        if float(np.linalg.norm(q3.forward_kinematics(*q_cand) - q3.OBS_CENTER)) >= q3.OBS_RADIUS + 0.002:
            break
        if float(np.dot(g, u)) < 0.0:
            u = u - (float(np.dot(g, u)) / gn2) * g
        u = np.clip(0.5 * u, u_lo, u_hi)
    else:
        u = np.zeros(2, dtype=np.float64)

    return u


class GaussianActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 6, act_dim: int = 2):
        super().__init__()
        feat_dim = obs_dim + 2  # append (target - ee)
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.6))

    def _feat(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x[..., 4:6] - x[..., 2:4]], dim=-1)

    def forward(self, x: torch.Tensor):
        f = self._feat(x)
        mean = torch.tanh(self.actor(f)) * q3.MAX_VEL
        std = self.log_std.clamp(-2.0, 0.4).exp()
        return mean, std, self.critic(f).squeeze(-1)


def train_a2c_variant(
    episodes: int,
    seed: int,
    mode: str,
    device: str = "cpu",
) -> dict:
    """mode in ('unshielded', 'soft_penalty', 'cbf_shield')."""
    assert mode in ("unshielded", "soft_penalty", "cbf_shield")
    torch.manual_seed(seed)
    env = q3.SterileArmEnv(mode="continuous", seed=seed)
    ac = GaussianActorCritic().to(device)
    opt_actor = torch.optim.Adam(list(ac.actor.parameters()) + [ac.log_std], lr=2e-3)
    opt_critic = torch.optim.Adam(ac.critic.parameters(), lr=4e-3)
    gamma, lam = 0.98, 0.95
    batch_min_steps = 256

    returns = []
    cum_limit_hits = 0
    cum_obs_hits = 0
    limit_curve, obs_curve = [], []
    b_obs, b_act, b_ret, b_adv = [], [], [], []

    def _flush_batch():
        if not b_obs:
            return
        st_all = torch.as_tensor(np.asarray(b_obs, dtype=np.float32), device=device)
        act_all = torch.as_tensor(np.asarray(b_act, dtype=np.float32), device=device)
        ret_t = torch.as_tensor(np.asarray(b_ret, dtype=np.float32), device=device)
        adv_np = np.asarray(b_adv, dtype=np.float32)
        adv_t = torch.as_tensor((adv_np - adv_np.mean()) / (adv_np.std() + 1e-8), device=device)

        mean_all, std_all, _ = ac(st_all)
        dist_all = torch.distributions.Normal(mean_all, std_all)
        logp_all = dist_all.log_prob(act_all).sum(-1)
        ent = dist_all.entropy().sum(-1).mean()
        actor_loss = -(logp_all * adv_t).mean() - 0.002 * ent
        opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(list(ac.actor.parameters()) + [ac.log_std], 1.0)
        opt_actor.step()

        for _ in range(3):
            _, _, val_all = ac(st_all)
            critic_loss = F.mse_loss(val_all, ret_t)
            opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(ac.critic.parameters(), 1.0)
            opt_critic.step()

        b_obs.clear()
        b_act.clear()
        b_ret.clear()
        b_adv.clear()

    for _ in range(episodes):
        s = env.reset()
        ep_obs, ep_act, ep_rew, ep_val = [], [], [], []
        ep_ret = 0.0
        done = False
        while not done:
            st = torch.as_tensor(s, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, std, val = ac(st)
                u_raw = torch.distributions.Normal(mean, std).sample()
                u_np = u_raw.squeeze(0).cpu().numpy()

            u_exec = q3.cbf_safety_shield(env.q, u_np) if mode == "cbf_shield" else u_np
            sn, r, done, info = env.step(u_exec)
            if mode == "soft_penalty":
                if info["limit_hit"]:
                    r -= 0.25
                if info["obs_hit"]:
                    r -= 0.50

            cum_limit_hits += int(info["limit_hit"])
            cum_obs_hits += int(info["obs_hit"])
            ep_obs.append(s)
            ep_act.append(u_np)
            ep_rew.append(r)
            ep_val.append(float(val.item()))
            s = sn
            ep_ret += r

        with torch.no_grad():
            next_v = 0.0 if info["success"] else float(ac(torch.as_tensor(s, device=device).unsqueeze(0))[2].item())
        T = len(ep_rew)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv = next_v if t == T - 1 else ep_val[t + 1]
            not_term = 0.0 if (t == T - 1 and info["success"]) else 1.0
            delta = ep_rew[t] + gamma * not_term * nv - ep_val[t]
            gae = delta + gamma * lam * not_term * gae
            adv[t] = gae
        ret = adv + np.asarray(ep_val, dtype=np.float32)

        b_obs.extend(ep_obs)
        b_act.extend(ep_act)
        b_ret.extend(ret.tolist())
        b_adv.extend(adv.tolist())

        if len(b_obs) >= batch_min_steps:
            _flush_batch()

        returns.append(ep_ret)
        limit_curve.append(cum_limit_hits)
        obs_curve.append(cum_obs_hits)

    _flush_batch()

    # Greedy eval over 20 episodes
    succ, eval_obs = [], []
    for k in range(20):
        s = env.reset(seed=2000 + seed * 50 + k)
        done = False
        oh = 0
        while not done:
            with torch.no_grad():
                u_np = ac(torch.as_tensor(s, device=device).unsqueeze(0))[0].squeeze(0).cpu().numpy()
            u_exec = q3.cbf_safety_shield(env.q, u_np) if mode == "cbf_shield" else u_np
            s, _, done, info = env.step(u_exec)
            oh += int(info["obs_hit"])
        succ.append(float(info["success"]))
        eval_obs.append(float(oh > 0))

    return {
        "returns": np.asarray(returns),
        "limit_curve": np.asarray(limit_curve),
        "obs_curve": np.asarray(obs_curve),
        "final_return": float(np.mean(returns[-50:])),
        "total_limit_hits": cum_limit_hits,
        "total_obs_hits": cum_obs_hits,
        "greedy_success": float(np.mean(succ)),
        "eval_obs_rate": float(np.mean(eval_obs)),
        "model": ac,
    }


def record_rollout(model: nn.Module, kind: str, seed: int, device: str = "cpu") -> list[dict]:
    mode = "discrete" if kind == "dqn" else "continuous"
    env = q3.SterileArmEnv(mode=mode, seed=seed)
    s = env.reset(seed=seed)
    frames = []
    lim_hits, obs_hits = 0, 0
    ee_trail = [q3.forward_kinematics(*env.q).copy()]
    frames.append({
        "q": env.q.copy(),
        "target": env.target.copy(),
        "trail": list(ee_trail),
        "step": 0,
        "dist": float(np.linalg.norm(ee_trail[-1] - env.target)),
        "lim_hits": 0,
        "obs_hits": 0,
        "success": False,
        "hit_now": False,
    })
    done = False
    while not done:
        with torch.no_grad():
            st = torch.as_tensor(s, device=device).unsqueeze(0)
            if kind == "dqn":
                action = int(model(st).argmax(1).item())
            else:
                u_np = model(st)[0].squeeze(0).cpu().numpy()
                action = q3.cbf_safety_shield(env.q, u_np) if kind == "cbf" else u_np
        s, _, done, info = env.step(action)
        lim_hits += int(info["limit_hit"])
        obs_hits += int(info["obs_hit"])
        ee_trail.append(q3.forward_kinematics(*env.q).copy())
        frames.append({
            "q": env.q.copy(),
            "target": env.target.copy(),
            "trail": list(ee_trail),
            "step": env.steps,
            "dist": info["dist"],
            "lim_hits": lim_hits,
            "obs_hits": obs_hits,
            "success": info["success"],
            "hit_now": bool(info["obs_hit"] or info["limit_hit"]),
        })
    return frames


def save_rollout_gif(
    dqn_model: nn.Module,
    unshielded_model: nn.Module,
    cbf_model: nn.Module,
    out_path: str,
    seed: int = 42,
    fps: int = 12,
) -> None:
    panels = [
        ("Double DQN (Discrete)", record_rollout(dqn_model, "dqn", seed)),
        ("Unshielded A2C (Continuous)", record_rollout(unshielded_model, "unshielded", seed)),
        ("Dual-Barrier CBF A2C", record_rollout(cbf_model, "cbf", seed)),
    ]
    n_frames = max(len(tr) for _, tr in panels)
    step_stride = max(1, n_frames // 90)
    frame_indices = list(range(0, n_frames, step_stride))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    lim_deg = float(np.rad2deg(q3.Q_MAX[0]))
    pil_frames = []
    for idx in frame_indices:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), dpi=100)
        for ax, (title, traj) in zip(axes, panels):
            fr = traj[min(idx, len(traj) - 1)]
            q1, q2 = float(fr["q"][0]), float(fr["q"][1])
            p0 = np.array([0.0, 0.0])
            p1 = np.array([q3.L1 * np.cos(q1), q3.L1 * np.sin(q1)])
            p2 = q3.forward_kinematics(q1, q2)

            ax.add_patch(patches.Wedge((0, 0), q3.L1 + q3.L2, -lim_deg, lim_deg, facecolor="#ECFDF5", edgecolor="#34D399", ls="--", lw=0.9, alpha=0.45))
            ax.add_patch(patches.Wedge((0, 0), q3.L1 + q3.L2, lim_deg, 360 - lim_deg, facecolor="#FEF2F2", edgecolor="#F87171", ls=":", lw=0.9, alpha=0.55))

            # Centrifuge keep-out disk
            obs_fc = "#EF4444" if fr["hit_now"] else "#FCA5A5"
            ax.add_patch(patches.Circle(q3.OBS_CENTER, q3.OBS_RADIUS, facecolor=obs_fc, edgecolor="#991B1B", lw=1.6, hatch="///", alpha=0.85, zorder=3))

            # Target vial
            ax.add_patch(patches.Circle(fr["target"], 0.06, facecolor="#BBF7D0", edgecolor="#15803D", lw=1.4, zorder=3))
            ax.scatter([fr["target"][0]], [fr["target"][1]], marker="x", s=50, color="#15803D", lw=1.8, zorder=4)

            tr_arr = np.asarray(fr["trail"])
            ax.plot(tr_arr[:, 0], tr_arr[:, 1], color="#6366F1", lw=1.4, alpha=0.75, zorder=4)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#1E3A8A", lw=5.0, solid_capstyle="round", zorder=5)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#2563EB", lw=4.0, solid_capstyle="round", zorder=5)
            ax.scatter([p0[0], p1[0]], [p0[1], p1[1]], s=50, color="#0F172A", zorder=6)
            ax.scatter([p2[0]], [p2[1]], s=65, color="#F59E0B", edgecolor="#0F172A", zorder=7)

            status = "SUCCESS" if fr["success"] else ("VIOLATION" if fr["hit_now"] else "RUNNING")
            st_col = "#15803D" if fr["success"] else ("#DC2626" if fr["hit_now"] else "#1E293B")
            hud = f"Step {fr['step']:3d} | Dist {fr['dist']:.3f}m\nLimHits: {fr['lim_hits']} | ObsHits: {fr['obs_hits']} | {status}"
            ax.text(0.02, 0.02, hud, transform=ax.transAxes, va="bottom", ha="left", fontsize=7.8,
                    fontweight="bold", color=st_col,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CBD5E1", alpha=0.92))
            ax.set_xlim(-1.12, 1.12)
            ax.set_ylim(-1.05, 1.12)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=9.5, fontweight="bold")
            ax.grid(alpha=0.22)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        pil_frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    durations = [int(1000 / fps)] * len(pil_frames)
    durations[-1] = 1500
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:], duration=durations, loop=0)
    print(f"Saved rollout animation -> {out_path} ({len(pil_frames)} frames, seed={seed})")


def patch_all() -> None:
    q3.cbf_safety_shield = cbf_safety_shield


def main() -> None:
    ap = argparse.ArgumentParser(description="CW1 Q3 reference solutions runner")
    ap.add_argument("--dqn-episodes", type=int, default=300)
    ap.add_argument("--a2c-episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fast", action="store_true", help="quick smoke-test settings (1 seed, 80 episodes)")
    ap.add_argument("--gif-seed", type=int, default=-1,
                    help="test episode seed for rollout GIF (-1 = auto-select a contrasting seed)")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    if args.fast:
        args.dqn_episodes = 80
        args.a2c_episodes = 80
        args.seeds = 1

    patch_all()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Part 1: DQN vs Double DQN vs No-Target-Net ────────────────────── #
    print("=" * 78)
    print(f"CW1 Q3.b/c — Discrete SterileArmEnv ({args.dqn_episodes} eps x {args.seeds} seed(s))")
    print("=" * 78)
    dqn_configs = [
        ("Standard DQN", False, True),
        ("Double DQN", True, True),
        ("No Target Net", False, False),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    saved_models = {}
    for name, dbl, tgt in dqn_configs:
        runs = [train_dqn_variant(args.dqn_episodes, s, double_dqn=dbl, use_target_net=tgt) for s in range(args.seeds)]
        saved_models[name] = runs[0]["model"]
        mean_ret = np.mean([r["returns"] for r in runs], axis=0)
        succ = np.mean([r["success_rate"] for r in runs]) * 100.0
        obs_r = np.mean([r["obs_collision_rate"] for r in runs]) * 100.0
        dist = np.mean([r["final_dist"] for r in runs])
        chat = np.mean([r["chattering"] for r in runs])
        print(f"  {name:<16} | Succ: {succ:5.1f}% | ObsHit: {obs_r:5.1f}% | Dist: {dist:.3f}m | Chatter: {chat:5.2f} rad")
        w = min(20, len(mean_ret))
        ax.plot(np.convolve(mean_ret, np.ones(w) / w, mode="valid"), label=name, lw=1.8)
    ax.set(xlabel="Episode", ylabel="Return (MA)", title="Q3.c SterileArmEnv (Discrete): DQN Variants")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, "q3_dqn_comparison.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p1}\n")

    # ── Part 2: Unshielded vs Soft-Penalty vs Dual-Barrier CBF A2C ────── #
    print("=" * 78)
    print(f"CW1 Q3.d/e — Continuous SterileArmEnv ({args.a2c_episodes} eps x {args.seeds} seed(s))")
    print("=" * 78)
    a2c_configs = [
        ("Unshielded A2C", "unshielded", "#DC2626"),
        ("Soft-Penalty A2C", "soft_penalty", "#F59E0B"),
        ("Dual-Barrier CBF A2C", "cbf_shield", "#16A34A"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for name, mode, col in a2c_configs:
        runs = [train_a2c_variant(args.a2c_episodes, s, mode=mode) for s in range(args.seeds)]
        saved_models[mode] = runs[0]["model"]
        mean_ret = np.mean([r["returns"] for r in runs], axis=0)
        mean_lim = np.mean([r["limit_curve"] for r in runs], axis=0)
        mean_obs = np.mean([r["obs_curve"] for r in runs], axis=0)
        f_ret = np.mean([r["final_return"] for r in runs])
        t_lim = np.mean([r["total_limit_hits"] for r in runs])
        t_obs = np.mean([r["total_obs_hits"] for r in runs])
        g_suc = np.mean([r["greedy_success"] for r in runs]) * 100.0
        print(f"  {name:<22} | Ret(last50): {f_ret:+7.1f} | LimHits: {t_lim:7.1f} | ObsHits: {t_obs:7.1f} | GreedySucc: {g_suc:5.1f}%")
        w = min(20, len(mean_ret))
        axes[0].plot(np.convolve(mean_ret, np.ones(w) / w, mode="valid"), label=name, color=col, lw=1.7)
        axes[1].plot(mean_lim, label=name, color=col, lw=1.7)
        axes[2].plot(mean_obs, label=name, color=col, lw=1.7)

    axes[0].set(xlabel="Episode", ylabel="Return (MA)", title="Episode Return")
    axes[1].set(xlabel="Episode", ylabel="Cumulative limit hits", title="Joint-Limit Violations")
    axes[2].set(xlabel="Episode", ylabel="Cumulative centrifuge hits", title="Centrifuge Collisions")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(args.out_dir, "q3_a2c_safety.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p2}")

    gif_seed = args.gif_seed
    if gif_seed < 0:
        gif_seed = 42
        fallback_seed = None
        for cand_seed in range(0, 300):
            u_tr = record_rollout(saved_models["unshielded"], "unshielded", cand_seed)
            c_tr = record_rollout(saved_models["cbf_shield"], "cbf", cand_seed)
            if len(c_tr) >= 12 and u_tr[-1]["obs_hits"] > 0 and c_tr[-1]["success"]:
                d_tr = record_rollout(saved_models["Double DQN"], "dqn", cand_seed)
                if d_tr[-1]["success"]:
                    gif_seed = cand_seed
                    break
                if fallback_seed is None:
                    fallback_seed = cand_seed
            elif fallback_seed is None and len(c_tr) >= 12 and (u_tr[-1]["obs_hits"] > 0 or u_tr[-1]["lim_hits"] > 0) and c_tr[-1]["success"]:
                fallback_seed = cand_seed
        else:
            if fallback_seed is not None:
                gif_seed = fallback_seed
        print(f"Auto-selected contrasting GIF seed: {gif_seed}")

    gif_path = os.path.join(args.out_dir, "q3_sterile_arm_rollout.gif")
    save_rollout_gif(
        saved_models["Double DQN"],
        saved_models["unshielded"],
        saved_models["cbf_shield"],
        gif_path,
        seed=gif_seed,
    )
    print("CW1_Q3_SOLUTIONS_DONE")


if __name__ == "__main__":
    main()
