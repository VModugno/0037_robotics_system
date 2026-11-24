"""COMP0037 2026/27 Coursework 1 — Q1 Starter Code: Hospital Charging/Dispatch Bandit.

MedBot must repeatedly choose among K = 8 hospital pharmacy dispatch/charging bays
whose throughput rewards are noisy and drift over a shift (non-stationary).
"""
from __future__ import annotations

import argparse
import numpy as np


class HospitalBayBandit:
    """K-armed Gaussian bandit modelling K = 8 hospital dispatch/charging bays.

    If `drift_std > 0`, true arm means undergo a Gaussian random walk at every
    step: mu_a <- mu_a + N(0, drift_std^2).
    """

    def __init__(self, k: int = 8, drift_std: float = 0.0, seed: int = 0) -> None:
        self.k = k
        self.drift_std = drift_std
        self.rng = np.random.default_rng(seed)
        self.means = self.rng.normal(0.0, 1.0, size=k)

    def pull(self, action: int) -> tuple[float, float, float, int]:
        """Pull bay `action`. Returns (reward, chosen_mean, optimal_mean, optimal_arm)."""
        best_arm = int(np.argmax(self.means))
        best_mean = float(self.means[best_arm])
        chosen_mean = float(self.means[action])
        reward = float(self.rng.normal(chosen_mean, 1.0))
        if self.drift_std > 0.0:
            self.means = self.means + self.rng.normal(0.0, self.drift_std, size=self.k)
        return reward, chosen_mean, best_mean, best_arm


def run_epsilon_greedy_sample_avg(
    k: int, steps: int, epsilon: float, drift_std: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Reference sample-average epsilon-greedy agent (COMPLETE)."""
    env = HospitalBayBandit(k=k, drift_std=drift_std, seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    q = np.zeros(k, dtype=np.float64)
    n = np.zeros(k, dtype=np.int64)
    regret = np.zeros(steps, dtype=np.float64)
    optimal = np.zeros(steps, dtype=np.float64)
    cum = 0.0
    for t in range(steps):
        if rng.random() < epsilon:
            a = int(rng.integers(k))
        else:
            a = int(np.argmax(q))
        r, chosen_mean, best_mean, best_arm = env.pull(a)
        n[a] += 1
        q[a] += (r - q[a]) / n[a]
        cum += best_mean - chosen_mean
        regret[t] = cum
        optimal[t] = float(a == best_arm)
    return regret, optimal


def run_ucb1(
    k: int, steps: int, c: float = 1.0, drift_std: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """TODO(CW1-Q1b): Implement UCB1 on HospitalBayBandit."""
    raise NotImplementedError("TODO(CW1-Q1b): implement UCB1")


def run_epsilon_greedy_const_alpha(
    k: int, steps: int, epsilon: float = 0.1, alpha: float = 0.1, drift_std: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """TODO(CW1-Q1c): Implement constant step-size epsilon-greedy (Q_a <- Q_a + alpha * (r - Q_a))."""
    raise NotImplementedError("TODO(CW1-Q1c): implement constant-alpha epsilon-greedy")


def main() -> None:
    ap = argparse.ArgumentParser(description="CW1 Q1: Hospital dispatch bay bandit")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--drift", type=float, default=0.0, help="random-walk std (0.0 stationary, 0.05 drifting)")
    args = ap.parse_args()

    regs, opts = [], []
    for run in range(args.runs):
        r, o = run_epsilon_greedy_sample_avg(8, args.steps, epsilon=0.1, drift_std=args.drift, seed=run)
        regs.append(r[-1])
        opts.append(o[-100:].mean() * 100.0)
    print(f"Sample-avg eps=0.1 (drift={args.drift}, T={args.steps}, {args.runs} runs): "
          f"mean regret={np.mean(regs):.2f}, final %opt={np.mean(opts):.1f}%")


if __name__ == "__main__":
    main()

