"""Instructor reference solutions for COMP0037 2026/27 Coursework 1 — Question 1.

Patches TODO(CW1-Q1b) (`run_ucb1`) and TODO(CW1-Q1c) (`run_epsilon_greedy_const_alpha`)
in `assessment/cw1_starter/q1_bandits_hospital.py` and runs:
  - Q1.b: Stationary 8-bay hospital bandit across horizons T in {200, 1000, 5000}
  - Q1.d: Non-stationary drifting bandit (drift_std = 0.05, T = 2000)

Usage (from repository root):
    .venv_rl\\Scripts\\python assessment\\cw1_solutions\\q1_bandits_solutions.py
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

import q1_bandits_hospital as q1  # noqa: E402


# --------------------------------------------------------------------------- #
# Reference implementations for Q1 TODOs
# --------------------------------------------------------------------------- #

def run_ucb1(
    k: int, steps: int, c: float = 1.0, drift_std: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Reference UCB1 agent on HospitalBayBandit."""
    env = q1.HospitalBayBandit(k=k, drift_std=drift_std, seed=seed)
    q = np.zeros(k, dtype=np.float64)
    n = np.zeros(k, dtype=np.int64)
    regret = np.zeros(steps, dtype=np.float64)
    optimal = np.zeros(steps, dtype=np.float64)
    cum = 0.0
    for t in range(steps):
        if t < k:
            a = t
        else:
            bonus = c * np.sqrt(np.log(t + 1) / n)
            a = int(np.argmax(q + bonus))
        r, chosen_mean, best_mean, best_arm = env.pull(a)
        n[a] += 1
        q[a] += (r - q[a]) / n[a]
        cum += best_mean - chosen_mean
        regret[t] = cum
        optimal[t] = float(a == best_arm)
    return regret, optimal


def run_epsilon_greedy_const_alpha(
    k: int,
    steps: int,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    drift_std: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference constant step-size epsilon-greedy agent."""
    env = q1.HospitalBayBandit(k=k, drift_std=drift_std, seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    q = np.zeros(k, dtype=np.float64)
    regret = np.zeros(steps, dtype=np.float64)
    optimal = np.zeros(steps, dtype=np.float64)
    cum = 0.0
    for t in range(steps):
        if rng.random() < epsilon:
            a = int(rng.integers(k))
        else:
            a = int(np.argmax(q))
        r, chosen_mean, best_mean, best_arm = env.pull(a)
        q[a] += alpha * (r - q[a])
        cum += best_mean - chosen_mean
        regret[t] = cum
        optimal[t] = float(a == best_arm)
    return regret, optimal


def patch_all() -> None:
    q1.run_ucb1 = run_ucb1
    q1.run_epsilon_greedy_const_alpha = run_epsilon_greedy_const_alpha


def eval_agent(fn, runs: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    regs, opts = [], []
    for seed in range(runs):
        r, o = fn(seed=seed, **kwargs)
        regs.append(r)
        opts.append(o)
    return np.mean(regs, axis=0), np.mean(opts, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="CW1 Q1 reference solutions runner")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    patch_all()
    os.makedirs(args.out_dir, exist_ok=True)
    k = 8

    # ── Q1.b: Stationary horizon sweep ────────────────────────────────── #
    print("=" * 72)
    print(f"CW1 Q1.b — Stationary 8-Bay Hospital Bandit ({args.runs} runs)")
    print("=" * 72)
    agents_stat = [
        ("eps=0.00 (greedy)", q1.run_epsilon_greedy_sample_avg, {"epsilon": 0.0}),
        ("eps=0.05", q1.run_epsilon_greedy_sample_avg, {"epsilon": 0.05}),
        ("eps=0.10", q1.run_epsilon_greedy_sample_avg, {"epsilon": 0.10}),
        ("eps=0.30", q1.run_epsilon_greedy_sample_avg, {"epsilon": 0.30}),
        ("UCB1 c=0.5", q1.run_ucb1, {"c": 0.5}),
        ("UCB1 c=1.0", q1.run_ucb1, {"c": 1.0}),
        ("UCB1 c=2.0", q1.run_ucb1, {"c": 2.0}),
    ]

    horizons = (200, 1000, 5000)
    header = f"{'Agent':<20}" + "".join(f" | T={T:<4} (Reg / %Opt)" for T in horizons)
    print(header)
    print("-" * len(header))

    curves_t5000 = {}
    for name, fn, kw in agents_stat:
        row = f"{name:<20}"
        for T in horizons:
            mean_reg, mean_opt = eval_agent(fn, args.runs, k=k, steps=T, drift_std=0.0, **kw)
            row += f" | {mean_reg[-1]:7.1f} / {mean_opt[-50:].mean()*100:5.1f}%"
            if T == 5000:
                curves_t5000[name] = (mean_reg, mean_opt)
        print(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (reg, opt) in curves_t5000.items():
        ax1.plot(reg, label=name)
        smooth_opt = np.convolve(opt * 100.0, np.ones(50) / 50.0, mode="valid")
        ax2.plot(smooth_opt, label=name)
    ax1.set(xlabel="Step", ylabel="Cumulative regret", title="Q1.b Stationary (T=5000): Regret")
    ax2.set(xlabel="Step", ylabel="% Optimal bay (50-step MA)", title="Q1.b Stationary (T=5000): Optimal Action %")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, "q1_stationary.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p1}\n")

    # ── Q1.d: Non-stationary drifting comparison ──────────────────────── #
    T_drift = 2000
    print("=" * 72)
    print(f"CW1 Q1.d — Non-Stationary Drifting Bandit (drift_std=0.05, T={T_drift}, {args.runs} runs)")
    print("=" * 72)
    agents_drift = [
        ("Sample-avg eps=0.1", q1.run_epsilon_greedy_sample_avg, {"epsilon": 0.1}),
        ("Const-alpha (a=0.1) eps=0.1", q1.run_epsilon_greedy_const_alpha, {"epsilon": 0.1, "alpha": 0.1}),
        ("UCB1 c=1.0", q1.run_ucb1, {"c": 1.0}),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, fn, kw in agents_drift:
        mean_reg, mean_opt = eval_agent(fn, args.runs, k=k, steps=T_drift, drift_std=0.05, **kw)
        print(f"  {name:<30} | Regret@{T_drift}: {mean_reg[-1]:8.1f} | Last-200 %Opt: {mean_opt[-200:].mean()*100:5.1f}%")
        ax1.plot(mean_reg, label=name, lw=1.8)
        smooth_opt = np.convolve(mean_opt * 100.0, np.ones(50) / 50.0, mode="valid")
        ax2.plot(smooth_opt, label=name, lw=1.8)
    ax1.set(xlabel="Step", ylabel="Cumulative regret", title="Q1.d Drifting Bandit (drift=0.05): Regret")
    ax2.set(xlabel="Step", ylabel="% Optimal bay (50-step MA)", title="Q1.d Drifting Bandit: Tracking Optimal Bay")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(args.out_dir, "q1_drifting.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p2}")
    print("CW1_Q1_SOLUTIONS_DONE")


if __name__ == "__main__":
    main()

