"""
COMP0037 — Week 02 Lab
=======================
Experiment driver for Q-Learning vs Sarsa (COMPLETE — no TODOs here).

Run me:
    pixi run -e week02 python gridworld.py    # environment self-test
    pixi run -e week02 python run_lab.py      # the experiments below

Produces:
    warehouse_learning.png  — learning curves on the stochastic warehouse
    cliff_learning.png      — learning curves on the cliff
    and console tables for the alpha sweep.
"""

from __future__ import annotations

import numpy as np

from agents import train
from gridworld import CliffEnv, WarehouseEnv


def moving_average(xs: list[float], k: int = 25) -> np.ndarray:
    return np.convolve(xs, np.ones(k) / k, mode="valid")


def main() -> None:
    import matplotlib.pyplot as plt

    # ── experiment 1: stochastic warehouse ─────────────────────────────
    env_w = WarehouseEnv(slip_prob=0.2)
    print("=" * 60)
    print("Experiment 1 — WarehouseEnv (slippery, gamma=0.95)")
    print("=" * 60)
    try:
        results = {}
        for method in ("q_learning", "sarsa"):
            Q, rets = train(env_w, method, episodes=600, alpha=0.2,
                            gamma=0.95, epsilon=0.1, seed=1)
            results[method] = (Q, rets)
            print(f"{method:>10}: last-100-episode mean return "
                  f"{np.mean(rets[-100:]):+.3f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        for method, (_, rets) in results.items():
            ax.plot(moving_average(rets), label=method)
        ax.set_xlabel("episode")
        ax.set_ylabel("return (25-episode moving average)")
        ax.set_title("WarehouseEnv — Q-Learning vs Sarsa (slip_prob=0.2)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("warehouse_learning.png", dpi=150)
        print("saved: warehouse_learning.png")
        plt.close(fig)
    except NotImplementedError as e:
        print(f"\n[!] {e}\n    Complete TODOs T1-T3 in agents.py first.")
        return

    # ── experiment 2: the cliff ────────────────────────────────────────
    env_c = CliffEnv()
    print("\n" + "=" * 60)
    print("Experiment 2 — CliffEnv (deterministic, gamma=1.0)")
    print("=" * 60)
    try:
        results = {}
        for method in ("q_learning", "sarsa"):
            Q, rets = train(env_c, method, episodes=500, alpha=0.5,
                            gamma=1.0, epsilon=0.1, seed=2)
            results[method] = (Q, rets)
            print(f"{method:>10}: last-100-episode mean return "
                  f"{np.mean(rets[-100:]):+.1f}")

        print("\nGreedy policies learned (U/D/L/R arrows; start is bottom-left; C cliff, G goal):")
        print(f"{'Q-Learning':^40s}{'Sarsa':^40s}")
        left = results["q_learning"][0]
        right = results["sarsa"][0]
        lines_l = env_c.render_policy(left).splitlines()
        lines_r = env_c.render_policy(right).splitlines()
        for ll, rr in zip(lines_l, lines_r):
            print(f"{ll}   {rr}")

        fig, ax = plt.subplots(figsize=(8, 5))
        for method, (_, rets) in results.items():
            ax.plot(moving_average(rets, 10), label=method)
        ax.set_xlabel("episode")
        ax.set_ylabel("return (10-episode moving average)")
        ax.set_title("CliffEnv — Q-Learning vs Sarsa (epsilon=0.1)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("cliff_learning.png", dpi=150)
        print("\nsaved: cliff_learning.png")
        plt.close(fig)
    except NotImplementedError as e:
        print(f"\n[!] {e}\n    Complete TODOs T1-T3 in agents.py first.")
        return

    # ── experiment 3: learning-rate sweep on the cliff ──────────────────
    print("\n" + "=" * 60)
    print("Experiment 3 — alpha sweep on CliffEnv (Q-Learning)")
    print("=" * 60)
    print(f"{'alpha':>8} | {'last-100 mean return':>20}")
    for alpha in (0.1, 0.3, 0.5, 0.7, 1.0):
        _, rets = train(env_c, "q_learning", episodes=300, alpha=alpha,
                        gamma=1.0, epsilon=0.1, seed=3)
        print(f"{alpha:>8.1f} | {np.mean(rets[-100:]):>+20.1f}")
    print("\n(Report Q3 asks you to explain the trend.)")


if __name__ == "__main__":
    main()
