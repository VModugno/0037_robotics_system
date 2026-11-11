"""
Instructor solutions for COMP0037 Week 01 (week01_bandits).

Reference implementations for every student TODO (T1-T7) are monkey-patched
onto the lab modules, then both lab mains run with default parameters to
reproduce the reference numbers in week01_bandits/LAB.md (self-check).

INSTRUCTOR USE ONLY -- students implement the TODOs themselves.

Notes:
* Implementations follow the lab docstrings/hints *literally* so that a
  hint-following student reproduces these exact numbers (the exact RNG call
  sequence matters). One documented deviation: the vectorised bandit
  exercise uses ``xp.random.randint`` because ``xp.random.integers`` does
  not exist in NumPy's legacy random API (the lab hint was fixed to match).
* Nothing is written into the lab folder: sys.dont_write_bytecode is set
  before the lab imports (no __pycache__ there), and the .png artifacts
  land in the current working directory.

Run from the labs/ root:
    pixi run --manifest-path pixi.toml -e week01 python solutions/week01_bandits_solutions.py
"""

import os
import sys
import time

LAB_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "week01_bandits")
)

os.environ.setdefault("MPLBACKEND", "Agg")   # headless-safe, no windows
sys.dont_write_bytecode = True               # never drop __pycache__ into the lab folder
sys.path.insert(0, LAB_DIR)

import matplotlib                            # noqa: E402
import numpy as np                           # noqa: E402
import mdp_formulation as mdp                # noqa: E402
import bandits                                # noqa: E402

N_ARMS = bandits.N_ARMS
BEST_ARM = bandits.BEST_ARM
BANDIT_PROBS = bandits.BANDIT_PROBS


# ============================================================ Part A (T1-T3)
def _reward(self, state, next_state):
    """T1: reward depends only on the landing cell (goal / mud / other)."""
    if next_state == self.goal:
        return self.REWARD_GOAL
    if next_state in self.mud:
        return self.REWARD_MUD
    return self.REWARD_STEP


def _step(self, state, action):
    """T2: stochastic transition kernel, exactly per the docstring steps:
    1. draw u ~ U(0,1) with self.rng
    2. if u < slip_prob, replace action with a uniformly chosen perpendicular
    3. landing cell via self._move
    4. reward via T1
    5. done iff landed on goal
    """
    u = self.rng.random()                        # 1. always draw
    if u < self.slip_prob:                        # 2. slip?
        perp = mdp.PERPENDICULAR[action]
        action = perp[int(self.rng.integers(2))]
    nxt = self._move(state, action)               # 3. deterministic move
    reward = self.reward(state, nxt)              # 4. T1
    done = nxt == self.goal                       # 5. terminal test
    return nxt, reward, done


def _expected_return(env, policy, n_episodes=300, seed=0):
    """T3: average discounted return over n_episodes independent episodes,
    each simulated with per-episode seed (seed + i) as the docstring hints."""
    total = 0.0
    for i in range(n_episodes):
        _, _, _, G = mdp.simulate_episode(env, policy, seed=seed + i)
        total += G
    return total / n_episodes


mdp.WarehouseRobot.reward = _reward
mdp.WarehouseRobot.step = _step
mdp.expected_return = _expected_return


# ============================================================ Part B (T4-T6)
def _bandit_update(self, arm, reward):
    """T5: incremental-mean update (running sample mean, O(1))."""
    self.N[arm] += 1
    self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


def _eps_greedy_select_action(self):
    """T4: explore with prob epsilon, else greedy — literal hint structure
    (one random() draw per step, then integers(N_ARMS) when exploring)."""
    if self.rng.random() < self.epsilon:
        return int(self.rng.integers(N_ARMS))
    return int(np.argmax(self.Q))


def _ucb_select_action(self):
    """T6: UCB1 — play an unpulled arm first (smallest N), else
    argmax_a Q[a] + c * sqrt(ln t / N[a]) with t = self.t."""
    if np.any(self.N == 0):
        return int(np.argmin(self.N))
    bonus = self.c * np.sqrt(np.log(self.t) / self.N)
    return int(np.argmax(self.Q + bonus))


bandits.BanditAgent.update = _bandit_update
bandits.EpsilonGreedy.select_action = _eps_greedy_select_action
bandits.UCB.select_action = _ucb_select_action
# UCB.update is COMPLETE in the scaffold: it delegates to BanditAgent.update
# (patched above) then increments self.t. OptimisticGreedy (epsilon=0.0,
# Q0=5) inherits the patched EpsilonGreedy.select_action.


# ============================================================ Part C (T7)
def _run_vectorized_epsilon_greedy(xp, epsilon, n_steps=1000, n_runs=20,
                                   seed=0):
    """T7: vectorised epsilon-greedy over n_runs problems simultaneously.

    The scaffold's COMPLETE parts are kept verbatim; the TODO(T7) block is
    filled per the docstring hint, except ``xp.random.integers`` (hint at
    bandits.py:291, nonexistent in NumPy's legacy random API) is replaced
    by ``xp.random.randint`` with identical semantics.
    """
    probs = xp.asarray(BANDIT_PROBS)              # (K,)
    regret_per_arm = probs.max() - probs          # (K,)
    n = n_runs
    rows = xp.arange(n)

    xp.random.seed(seed)
    U = xp.random.random((n, N_ARMS, n_steps))    # pre-sampled Bernoulli draws

    Q = xp.zeros((n, N_ARMS))
    N = xp.zeros((n, N_ARMS))  # float on purpose — we only ever add 1 to it
    step_regret = xp.zeros(n_steps)
    step_optimal = xp.zeros(n_steps)

    for t in range(n_steps):
        u = xp.random.random((n,))                        # uniform draws
        explore = u < epsilon                              # boolean mask
        greedy = xp.argmax(Q, axis=1)                      # shape (n,)
        random_arms = xp.random.randint(0, N_ARMS, n)      # shape (n,)
        arms = xp.where(explore, random_arms, greedy)

        rewards = (U[rows, arms, t] < probs[arms]).astype(xp.float64)

        # Vectorised incremental-mean update — the T5 twin
        n_a = N[rows, arms] + 1.0
        Q[rows, arms] += (rewards - Q[rows, arms]) / n_a
        N[rows, arms] = n_a

        step_regret[t] = xp.sum(regret_per_arm[arms])
        step_optimal[t] = xp.mean(arms == BEST_ARM)

    cum_regret = xp.cumsum(step_regret) / n
    return cum_regret


bandits.run_vectorized_epsilon_greedy = _run_vectorized_epsilon_greedy


# ============================================================ driver
def main():
    print("week01_bandits_solutions.py — instructor solutions run for week01_bandits")
    print(f"cwd: {os.getcwd()}")
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | "
          f"matplotlib {matplotlib.__version__}")
    print(f"hasattr(numpy.random, 'integers') = "
          f"{hasattr(np.random, 'integers')}   "
          f"(bandits.py:291 hints xp.random.integers)")
    for name in ("mdp_formulation.py", "bandits.py"):
        with open(os.path.join(LAB_DIR, name), encoding="utf-8") as fh:
            src = fh.read()
        print(f"TODO guards still present in {name}: "
              f"{src.count('TODO(T')} (lab file untouched, unsolved)")
    print()

    print("=" * 72)
    print("RUN 1/3: mdp_formulation.main()  (defaults: n_episodes=300, seed=0)")
    print("=" * 72)
    t0 = time.perf_counter()
    mdp.main()
    dt_a = time.perf_counter() - t0

    print()
    print("=" * 72)
    print("RUN 2/3: bandits.main()  (defaults: --steps 1000 --runs 20)")
    print("=" * 72)
    sys.argv = [sys.argv[0]]        # force argparse defaults
    t0 = time.perf_counter()
    bandits.main()
    dt_b = time.perf_counter() - t0

    print()
    print("=" * 72)
    print("SUPPLEMENTARY — final % optimal action per agent")
    print("(NOT printed by the scaffold main; same seeded experiment,")
    print(" deterministic, so identical to what main just computed)")
    print("=" * 72)
    t0 = time.perf_counter()
    for label, factory in bandits.AGENTS:
        regret, optimal = bandits.run_experiment(
            factory, n_steps=1000, n_runs=20
        )
        print(f"{label:20s}  % optimal @ t=1000: {optimal[-1] * 100:5.1f}%"
              f"   (regret {regret[-1]:7.2f})")
    dt_s = time.perf_counter() - t0

    print()
    print("=" * 72)
    print("PART C sanity check (pure NumPy CPU — NOT the --bench benchmark)")
    print("=" * 72)
    t0 = time.perf_counter()
    cr = bandits.run_vectorized_epsilon_greedy(
        np, 0.1, n_steps=1000, n_runs=20, seed=0
    )
    dt_c = time.perf_counter() - t0
    print(f"vectorised eps=0.1 (numpy, n_runs=20, n_steps=1000, seed=0):")
    print(f"  mean cumulative pseudo-regret @ t=1000: {float(cr[-1]):.2f}")

    print()
    print(f"runtimes: mdp main {dt_a:.2f}s | bandits main {dt_b:.2f}s | "
          f"supplementary {dt_s:.2f}s | T7 sanity {dt_c:.2f}s")


if __name__ == "__main__":
    main()
