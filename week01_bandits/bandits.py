"""
COMP0037 — Week 01 Lab, Part B
================================
Multi-armed bandits: epsilon-greedy vs UCB, and regret analysis.

YOUR TASKS (see LAB.md):
  * T4 — `EpsilonGreedy.select_action`   (explore vs exploit)
  * T5 — `BanditAgent.update`            (incremental-mean value estimate)
  * T6 — `UCB.select_action`            (upper-confidence-bound rule)
  * T7 — `run_vectorized_epsilon_greedy` (fully vectorised batch version)

Run me:
    pixi run -e week01 python bandits.py            # experiments + plots
    pixi run -e week01 python bandits.py --bench    # CPU vs GPU timing (Part C)
    pixi run -e gpu      python bandits.py --bench --gpu   # with CuPy on a CUDA GPU

GPU-ready NumPy (Part C): the same epsilon-greedy experiment, fully
vectorised over `n_runs` bandit problems at once, written against an array
module that can be NumPy (CPU) or CuPy (GPU) — measure when the GPU actually
wins.
"""

from __future__ import annotations

import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------ bandit
# True (hidden) success probabilities of the 10 arms. The agent never sees
# these — it only observes pulls.
BANDIT_PROBS = np.array(
    [0.10, 0.35, 0.60, 0.45, 0.80, 0.25, 0.90, 0.50, 0.70, 0.15]
)
N_ARMS = len(BANDIT_PROBS)
BEST_ARM = int(np.argmax(BANDIT_PROBS))  # arm 6, p=0.90


class BernoulliBandit:
    """A stationary K-armed Bernoulli bandit (COMPLETE — read it)."""

    def __init__(self, probs: np.ndarray, seed: int = 0) -> None:
        self.probs = np.asarray(probs, dtype=float)
        self.rng = np.random.default_rng(seed)

    def pull(self, arm: int) -> float:
        """Draw a reward ~ Bernoulli(p[arm]); 1.0 = success, 0.0 = failure."""
        return 1.0 if self.rng.random() < self.probs[arm] else 0.0

    def regret(self, arm: int) -> float:
        """Pseudo-regret of pulling `arm`: the gap to the best arm's mean."""
        return float(self.probs.max() - self.probs[arm])


# ------------------------------------------------------------------ agents
class BanditAgent:
    """Base class for bandit algorithms.

    State:
      Q — estimated value of each arm,      shape (K,)
      N — number of times each arm pulled, shape (K,)
    """

    def __init__(self, n_arms: int, rng: np.random.Generator) -> None:
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms, dtype=np.int64)
        self.rng = rng

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        """Incremental-mean value update (TODO T5).

        After n pulls of an arm with observed rewards, Q should be the
        *sample mean* of those rewards. Do NOT store the reward history —
        update Q in O(1) with the running-mean formula:

            N[arm]  += 1
            Q[arm]  += (reward - Q[arm]) / N[arm]

        This one function is shared by every agent in this file.
        """
        raise NotImplementedError(
            "TODO(T5): implement BanditAgent.update — see LAB.md Part B."
        )


class EpsilonGreedy(BanditAgent):
    """epsilon-greedy: explore with probability epsilon, else exploit (TODO T4)."""

    def __init__(self, n_arms: int, epsilon: float, rng: np.random.Generator,
                 q_init: float = 0.0) -> None:
        super().__init__(n_arms, rng)
        self.Q.fill(q_init)
        self.epsilon = epsilon

    def select_action(self) -> int:
        """TODO(T4):

          * with probability `self.epsilon`: return a uniformly random arm
            (hint: ``self.rng.random() < self.epsilon``, then
            ``self.rng.integers(N_ARMS)``)
          * otherwise: return the greedy arm  ``argmax self.Q``
        """
        raise NotImplementedError(
            "TODO(T4): implement EpsilonGreedy.select_action — see LAB.md Part B."
        )


class OptimisticGreedy(EpsilonGreedy):
    """Greedy (epsilon=0) but with optimistic initial values Q0=+5 (COMPLETE).

    Pure exploitation of wildly optimistic estimates forces the agent to try
    every arm early — exploration that *falls out of the value estimates*,
    with no randomness at all.
    """

    def __init__(self, n_arms: int, rng: np.random.Generator) -> None:
        super().__init__(n_arms, epsilon=0.0, rng=rng, q_init=5.0)


class UCB(BanditAgent):
    """Upper Confidence Bound action selection (TODO T6)."""

    def __init__(self, n_arms: int, c: float, rng: np.random.Generator,
                 q_init: float = 0.0) -> None:
        super().__init__(n_arms, rng)
        self.Q.fill(q_init)
        self.c = c
        self.t = 0  # global pull counter

    def select_action(self) -> int:
        """TODO(T6): the UCB1 rule.

          * if any arm has never been pulled (N[a] == 0), play such an arm
            first (e.g. the one with the smallest N),
          * otherwise play   argmax_a  Q[a] + c * sqrt( ln t / N[a] )

        `t` is the number of pulls so far (self.t). The bonus shrinks as an
        arm is sampled more (you become certain about it) and grows with the
        total time (uncertainty about rarely-tried arms grows).
        """
        raise NotImplementedError(
            "TODO(T6): implement UCB.select_action — see LAB.md Part B."
        )

    def update(self, arm: int, reward: float) -> None:
        """Same incremental mean as the base class, plus the time counter."""
        BanditAgent.update(self, arm, reward)  # once T5 is done this works
        self.t += 1


# ------------------------------------------------------------ experiments
def run_experiment(agent_factory, n_steps: int = 1000, n_runs: int = 20,
                   base_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Average learning behaviour of one algorithm over `n_runs` bandit problems.

    Returns (avg_cumulative_regret[n_steps], frac_optimal_action[n_steps]).
    COMPLETE — read it to see how your agents are exercised.
    """
    per_arm_regret = BANDIT_PROBS.max() - BANDIT_PROBS  # (K,)
    cum_regret = np.zeros(n_steps)
    optimal = np.zeros(n_steps)
    for run in range(n_runs):
        bandit = BernoulliBandit(BANDIT_PROBS, seed=base_seed + run)
        agent = agent_factory(np.random.default_rng(base_seed + 1000 + run))
        run_regret = 0.0
        for t in range(n_steps):
            a = agent.select_action()
            r = bandit.pull(a)
            agent.update(a, r)
            run_regret += per_arm_regret[a]
            cum_regret[t] += run_regret
            optimal[t] += float(a == BEST_ARM)
    return cum_regret / n_runs, optimal / n_runs


AGENTS = [
    ("eps=0.0 (greedy)", lambda rng: EpsilonGreedy(N_ARMS, 0.0, rng)),
    ("eps=0.01", lambda rng: EpsilonGreedy(N_ARMS, 0.01, rng)),
    ("eps=0.1", lambda rng: EpsilonGreedy(N_ARMS, 0.1, rng)),
    ("eps=0.5", lambda rng: EpsilonGreedy(N_ARMS, 0.5, rng)),
    ("UCB c=0.5", lambda rng: UCB(N_ARMS, 0.5, rng)),
    ("UCB c=1.0", lambda rng: UCB(N_ARMS, 1.0, rng)),
    ("UCB c=2.0", lambda rng: UCB(N_ARMS, 2.0, rng)),
    ("optimistic Q0=5", lambda rng: OptimisticGreedy(N_ARMS, rng)),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Week 01 bandit experiments")
    ap.add_argument("--bench", action="store_true",
                    help="run Part C (vectorised CPU vs GPU timing benchmark)")
    ap.add_argument("--gpu", action="store_true",
                    help="use CuPy for the vectorised benchmark (needs the `gpu` pixi env)")
    ap.add_argument("--show", action="store_true", help="also open plot windows")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    if args.bench:
        bench(use_gpu=args.gpu)
        return

    print(f"10-armed Bernoulli bandit | best arm = {BEST_ARM} "
          f"(p*={BANDIT_PROBS[BEST_ARM]:.2f}) | {args.steps} steps, {args.runs} runs\n")

    results = {}
    for label, factory in AGENTS:
        try:
            regret, optimal = run_experiment(
                factory, n_steps=args.steps, n_runs=args.runs
            )
            results[label] = (regret, optimal)
            print(f"{label:20s}  cumulative regret @ t={args.steps}: {regret[-1]:7.2f}")
        except NotImplementedError as e:
            print(f"\n[!] {e}")
            print("    Complete TODOs T4-T6, then re-run this script.")
            return

    if not results:
        return

    # ── figure 1: cumulative pseudo-regret ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (regret, _) in results.items():
        ax.plot(regret, label=label, linewidth=1.8)
    ax.set_xlabel("time step t")
    ax.set_ylabel("average cumulative pseudo-regret  (lower = better)")
    ax.set_title("10-armed Bernoulli bandit — exploration strategies")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("regret_comparison.png", dpi=150)
    print(f"\nsaved: regret_comparison.png")

    # ── figure 2: fraction of pulls on the best arm ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (_, optimal) in results.items():
        ax.plot(optimal, label=label, linewidth=1.8)
    ax.set_xlabel("time step t")
    ax.set_ylabel("fraction of runs pulling the best arm")
    ax.set_title("% optimal action")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("optimal_action.png", dpi=150)
    print("saved: optimal_action.png")

    if args.show:
        plt.show()
    else:
        plt.close("all")


# ─────────────────────────────────────────────────────────── Part C ──────
# The same epsilon-greedy experiment, fully vectorised over `n_runs`
# problems at once, so the array math can run on NumPy (CPU) *or* CuPy (GPU).
# `xp` is whichever array module we were given — the code is identical.


def get_xp(use_gpu: bool):
    """Return the array module to use: CuPy if requested and available, else NumPy."""
    if use_gpu:
        try:
            import cupy as cp  # noqa: PLC0415

            return cp, "cupy (GPU)"
        except ImportError:
            print("[!] CuPy not available — falling back to NumPy.")
            print("    For GPU runs: pixi run -e gpu python bandits.py --bench --gpu")
    return np, "numpy (CPU)"


def run_vectorized_epsilon_greedy(xp, epsilon: float, n_steps: int = 1000,
                                  n_runs: int = 20, seed: int = 0) -> np.ndarray:
    """Vectorised epsilon-greedy over `n_runs` bandit problems simultaneously.

    Every array below carries a leading `n_runs` axis, so one python loop
    over time-steps advances ALL problems at once. `xp` is NumPy or CuPy.

    TODO(T7): produce the batch of arm choices for this time step:

        u           = xp.random.random((n_runs,))          # uniform draws
        explore     = u < epsilon                           # boolean mask
        greedy      = argmax of Q over arms (axis 1)       # shape (n_runs,)
        random_arms = xp.random.randint(0, N_ARMS, n_runs)
        arms        = xp.where(explore, random_arms, greedy)

    Returns the mean cumulative pseudo-regret over the runs, shape (n_steps,).
    """
    probs = xp.asarray(BANDIT_PROBS)  # (K,)
    regret_per_arm = probs.max() - probs  # (K,)
    n = n_runs
    rows = xp.arange(n)

    # Pre-sample all rewards: U[r, a, t] < p[a]  <=>  Bernoulli success.
    # (Memory: n_runs * K * n_steps floats — small enough here, and it lets
    # us time the *agent* rather than the random number generator.)
    xp.random.seed(seed)
    U = xp.random.random((n, N_ARMS, n_steps))

    Q = xp.zeros((n, N_ARMS))
    N = xp.zeros((n, N_ARMS))  # float on purpose — we only ever add 1 to it
    step_regret = xp.zeros(n_steps)

    for t in range(n_steps):
        arms = xp.zeros(n, dtype=xp.int64)  # ← replace with TODO(T7) result

        # ── TODO(T7): implement the vectorised action selection here ─────
        raise NotImplementedError(
            "TODO(T7): implement run_vectorized_epsilon_greedy — see LAB.md Part C."
        )
        # ──────────────────────────────────────────────────────────────────

        rewards = (U[rows, arms, t] < probs[arms]).astype(xp.float64)

        # Vectorised incremental-mean update — compare with your T5!
        n_a = N[rows, arms] + 1.0
        Q[rows, arms] += (rewards - Q[rows, arms]) / n_a
        N[rows, arms] = n_a

        step_regret[t] = xp.sum(regret_per_arm[arms])

    cum_regret = xp.cumsum(step_regret) / n
    return cum_regret


def bench(use_gpu: bool) -> None:
    """Measure when 'GPU-ready numpy' actually pays off (COMPLETE).

    Runs the vectorised experiment at a small and a large batch size and
    prints throughput. Discuss the numbers in your report — the GPU is not
    automatically faster; per-step launch overhead dominates small problems.
    """
    xp, name = get_xp(use_gpu)
    print(f"array backend: {name}\n")
    print(f"{'n_runs':>8} | {'total pulls/s':>14} | wall-clock")
    print("-" * 44)
    for n_runs in (20, 5000):
        t0 = time.perf_counter()
        run_vectorized_epsilon_greedy(xp, 0.1, n_steps=1000, n_runs=n_runs)
        if name.startswith("cupy"):
            xp.cuda.Stream.null.synchronize()  # CuPy is async — time it fairly
        dt = time.perf_counter() - t0
        print(f"{n_runs:>8} | {n_runs * 1000 / dt:>14,.0f} | {dt:.2f}s")


if __name__ == "__main__":
    main()
