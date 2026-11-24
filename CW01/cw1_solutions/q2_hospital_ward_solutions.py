"""Instructor reference solutions for COMP0037 2026/27 Coursework 1 — Question 2.

Patches TODO(CW1-Q2b) (`train_tabular`) in `assessment/cw1_starter/q2_hospital_ward.py`
and executes the controlled sweeps on `HospitalWardEnv` (8x12, start=(7,0), goal=(0,11)):
  - Wheel slip sweep: p_slip in {0.0, 0.1, 0.2, 0.4} at epsilon = 0.1
  - Exploration sweep: epsilon in {0.05, 0.1, 0.25} at p_slip = 0.2

Usage (from repository root):
    .venv_rl\\Scripts\\python -X utf8 assessment\\cw1_solutions\\q2_hospital_ward_solutions.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
STARTER_DIR = os.path.normpath(os.path.join(CUR_DIR, os.pardir, "cw1_starter"))
OUT_DIR = os.path.join(CUR_DIR, "outputs")
sys.path.insert(0, STARTER_DIR)

import q2_hospital_ward as q2  # noqa: E402


# --------------------------------------------------------------------------- #
# Reference implementation for Q2 TODO
# --------------------------------------------------------------------------- #

def select_eps_greedy(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    return int(np.argmax(q_row))


def train_tabular(
    env: q2.HospitalWardEnv,
    method: str,
    episodes: int = 800,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    seed: int = 0,
    max_steps: int = 300,
) -> tuple[np.ndarray, list[float]]:
    """Reference Q-Learning ('q_learning') and SARSA ('sarsa') on HospitalWardEnv."""
    assert method in ("q_learning", "sarsa")
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    returns: list[float] = []

    for _ in range(episodes):
        s = env.reset()
        a = select_eps_greedy(q[s], epsilon, rng)
        ep_ret = 0.0
        for _ in range(max_steps):
            s_next, r, done = env.step(a)
            ep_ret += r
            if method == "q_learning":
                target = r if done else r + gamma * float(np.max(q[s_next]))
                q[s, a] += alpha * (target - q[s, a])
                s = s_next
                a = select_eps_greedy(q[s], epsilon, rng)
            else:
                a_next = select_eps_greedy(q[s_next], epsilon, rng)
                target = r if done else r + gamma * float(q[s_next, a_next])
                q[s, a] += alpha * (target - q[s, a])
                s, a = s_next, a_next
            if done:
                break
        returns.append(ep_ret)
    return q, returns


def moving_average(xs: list[float], w: int = 30) -> np.ndarray:
    arr = np.asarray(xs, dtype=np.float64)
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def patch_all() -> None:
    q2.train_tabular = train_tabular


def main() -> None:
    ap = argparse.ArgumentParser(description="CW1 Q2 reference solutions runner")
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    patch_all()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Sweep 1: p_slip in {0.0, 0.1, 0.2, 0.4} at eps=0.1 ────────────── #
    slips = (0.0, 0.1, 0.2, 0.4)
    print("=" * 72)
    print(f"CW1 Q2.b — Wheel Slip Sweep on HospitalWardEnv (eps=0.1, {args.episodes} eps)")
    print("=" * 72)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    for ax, slip in zip(axes.flat, slips):
        for method, col in (("q_learning", "#2563EB"), ("sarsa", "#DC2626")):
            env = q2.HospitalWardEnv(slip_prob=slip, seed=0)
            q_tab, rets = q2.train_tabular(env, method, episodes=args.episodes, epsilon=0.1, seed=1)
            print(f"  slip={slip:.1f} | {method:<10} | mean return (last 100): {np.mean(rets[-100:]):+7.2f}")
            ax.plot(moving_average(rets), label=method, color=col, lw=1.6)
            if slip in (0.0, 0.2):
                print(f"\n  Greedy policy ({method}, slip={slip:.1f}):")
                for line in env.render_policy(q_tab).splitlines():
                    print("    " + line)
                print()
        ax.set_title(f"slip_prob = {slip:.1f} (eps = 0.1)")
        ax.set_ylabel("Episode return (30-ep MA)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, "q2_slip_sweep.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p1}\n")

    # ── Sweep 2: epsilon in {0.05, 0.1, 0.25} at p_slip=0.2 ───────────── #
    epsilons = (0.05, 0.1, 0.25)
    print("=" * 72)
    print(f"CW1 Q2.b — Exploration Sweep on HospitalWardEnv (slip=0.2, {args.episodes} eps)")
    print("=" * 72)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    for ax, eps in zip(axes, epsilons):
        for method, col in (("q_learning", "#2563EB"), ("sarsa", "#DC2626")):
            env = q2.HospitalWardEnv(slip_prob=0.2, seed=0)
            _, rets = q2.train_tabular(env, method, episodes=args.episodes, epsilon=eps, seed=2)
            print(f"  eps={eps:.2f} | {method:<10} | mean return (last 100): {np.mean(rets[-100:]):+7.2f}")
            ax.plot(moving_average(rets), label=method, color=col, lw=1.6)
        ax.set_title(f"epsilon = {eps:.2f} (slip = 0.2)")
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Episode return (30-ep MA)")
    fig.tight_layout()
    p2 = os.path.join(args.out_dir, "q2_eps_sweep.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p2}")
    print("CW1_Q2_SOLUTIONS_DONE")


if __name__ == "__main__":
    main()

