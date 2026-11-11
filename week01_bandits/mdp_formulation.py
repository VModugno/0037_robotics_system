"""
COMP0037 — Week 01 Lab, Part A
================================
Formalising a robotics task as a Markov Decision Process (MDP).

You will complete a small gridworld ("warehouse robot") so that it fully
implements the MDP tuple (S, A, T, R, gamma), then use Monte-Carlo simulation
to estimate the expected return of different policies.

YOUR TASKS (see LAB.md for the full brief):
  * T1 — implement `WarehouseRobot.reward`
  * T2 — implement `WarehouseRobot.step`  (the stochastic transition kernel T)
  * T3 — implement `expected_return`      (Monte-Carlo evaluation of a policy)

Run me:
    pixi run -e week01 python mdp_formulation.py
    (or, on Colab / with plain Python:  python mdp_formulation.py)

On the first run you will see a friendly NotImplementedError — that is your
starting point. Read LAB.md, then fill in the TODOs.
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ actions
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["U", "R", "D", "L"]
N_ACTIONS = 4

# (row_delta, col_delta) of the *intended* move for each action.
DELTA: dict[int, tuple[int, int]] = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

# The two perpendicular actions the robot may "slip" into instead.
PERPENDICULAR: dict[int, tuple[int, int]] = {
    UP: (LEFT, RIGHT),
    RIGHT: (UP, DOWN),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
}


class WarehouseRobot:
    """A delivery robot navigating a small warehouse grid.

    The MDP tuple (S, A, T, R, gamma):

    * S — every grid cell (row, col) that is not a shelf '#'.
    * A — {UP, RIGHT, DOWN, LEFT} (4 discrete actions).
    * T — the robot executes the *intended* action with probability
      ``1 - slip_prob``; with probability ``slip_prob`` it slips and executes
      one of the two perpendicular actions instead (each with probability
      ``slip_prob / 2``). Moves into a shelf or off the grid leave the robot
      where it was (it bumps into the obstacle).
    * R — +1.0 on reaching the goal, -0.2 for landing on mud, -0.04 time
      cost for any other step. Rewards depend only on the landing cell.
    * gamma — discount factor (default 0.95).

    Read this class top-to-bottom before writing any code: everything you
    need to know about the mechanics of this little world is written here.
    """

    # '.' free, '#' shelf (wall), 'm' mud, 'R' robot start, 'G' goal
    LAYOUT = [
        "..........",
        "...##.....",
        "..........",
        "..##..mm..",
        ".....mm...",
        "R.......G.",
    ]

    REWARD_GOAL = 1.0
    REWARD_MUD = -0.2
    REWARD_STEP = -0.04

    def __init__(
        self,
        slip_prob: float = 0.2,
        gamma: float = 0.95,
        seed: int | None = 0,
    ) -> None:
        self.n_rows = len(self.LAYOUT)
        self.n_cols = len(self.LAYOUT[0])
        self.slip_prob = slip_prob
        self.gamma = gamma

        self.shelves: set[tuple[int, int]] = set()
        self.mud: set[tuple[int, int]] = set()
        self.start: tuple[int, int] = (0, 0)
        self.goal: tuple[int, int] = (0, 0)
        for r, row in enumerate(self.LAYOUT):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.shelves.add((r, c))
                elif ch == "m":
                    self.mud.add((r, c))
                elif ch == "R":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------- queries
    def reset(self, seed: int | None = None) -> tuple[int, int]:
        """Return the start state; reseed the RNG only if `seed` is given (COMPLETE)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.start

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def is_free(self, cell: tuple[int, int]) -> bool:
        return self.in_bounds(cell) and cell not in self.shelves

    def states(self) -> list[tuple[int, int]]:
        """All non-shelf cells — the state space S (COMPLETE)."""
        return [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if (r, c) not in self.shelves
        ]

    def n_states(self) -> int:
        return len(self.states())

    def render(self, state: tuple[int, int]) -> str:
        """ASCII view of the warehouse with the robot at `state` (COMPLETE)."""
        lines = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                if (r, c) == state:
                    row.append("R")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) in self.shelves:
                    row.append("#")
                elif (r, c) in self.mud:
                    row.append("m")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)

    # ------------------------------------------------------------ dynamics
    def _move(self, state: tuple[int, int], action: int) -> tuple[int, int]:
        """Deterministic move: the intended cell, or stay put if blocked (COMPLETE)."""
        dr, dc = DELTA[action]
        nxt = (state[0] + dr, state[1] + dc)
        if self.is_free(nxt):
            return nxt
        return state  # bumped into a shelf or the outer wall

    def reward(self, state: tuple[int, int], next_state: tuple[int, int]) -> float:
        """Immediate reward for *landing* on `next_state` (TODO T1).

        Rules (see the class docstring):
          * landing on the goal            -> REWARD_GOAL   (+1.0)
          * landing on mud                -> REWARD_MUD    (-0.2)
          * anything else (incl. bumping) -> REWARD_STEP   (-0.04)
        """
        raise NotImplementedError(
            "TODO(T1): implement WarehouseRobot.reward — see LAB.md Part A."
        )

    def step(
        self, state: tuple[int, int], action: int
    ) -> tuple[tuple[int, int], float, bool]:
        """One stochastic transition (TODO T2).

        This function IS the transition kernel T of the MDP:
          1. Draw u ~ U(0, 1) with ``self.rng``.
          2. If u < self.slip_prob, replace `action` with one of the two
             perpendicular actions, chosen uniformly with ``self.rng``.
          3. Compute the landing cell with ``self._move`` (COMPLETE helper).
          4. Compute the reward with ``self.reward`` (your T1).
          5. The episode ends (`done=True`) only when the robot lands on the goal.

        Returns (next_state, reward, done).
        """
        raise NotImplementedError(
            "TODO(T2): implement WarehouseRobot.step — see LAB.md Part A."
        )


# ------------------------------------------------------------- policies
# A policy maps (env, state, rng) -> action index. Two are provided.


def random_policy(env: WarehouseRobot, state: tuple[int, int], rng: np.random.Generator) -> int:
    """Uniformly random behaviour (COMPLETE)."""
    return int(rng.integers(N_ACTIONS))


def hand_crafted_policy(env: WarehouseRobot, state: tuple[int, int], rng: np.random.Generator) -> int:
    """A hand-written 'head right, then down' delivery policy (COMPLETE).

    Written by a human who can see the whole map — you will compare it
    against the random policy and against what RL can learn later.
    """
    r, c = state
    gr, gc = env.goal
    if c < gc:
        return RIGHT
    if r < gr:
        return DOWN
    if c > gc:
        return LEFT
    return UP


# ------------------------------------------------------------ simulation
MAX_STEPS = 500  # safety cap so broken policies cannot loop forever


def simulate_episode(
    env: WarehouseRobot,
    policy,
    seed: int | None = 0,
) -> tuple[list[tuple[int, int]], list[int], list[float], float]:
    """Roll out one episode under `policy` (COMPLETE).

    Returns (states, actions, rewards, G) where G is the discounted return
        G = r_0 + gamma * r_1 + gamma^2 * r_2 + ...
    """
    env.reset(seed)
    rng = np.random.default_rng(None if seed is None else seed + 1)
    s = env.start
    states, actions, rewards = [s], [], []
    G, discount = 0.0, 1.0
    for _ in range(MAX_STEPS):
        a = policy(env, s, rng)
        s2, r, done = env.step(s, a)
        actions.append(a)
        rewards.append(r)
        G += discount * r
        discount *= env.gamma
        s = s2
        states.append(s)
        if done:
            break
    return states, actions, rewards, G


def expected_return(env: WarehouseRobot, policy, n_episodes: int = 300, seed: int = 0) -> float:
    """Monte-Carlo estimate of the expected discounted return of `policy` (TODO T3).

    Simulate ``n_episodes`` independent episodes and return the *average*
    discounted return. Use a different RNG seed per episode (e.g. seed + i)
    so the episodes are independent but the experiment is reproducible.

    Hint: everything you need is in ``simulate_episode`` above.
    """
    raise NotImplementedError(
        "TODO(T3): implement expected_return — see LAB.md Part A."
    )


# ------------------------------------------------------------------ main
def main() -> None:
    env = WarehouseRobot()
    print("Warehouse map (R = robot, G = goal, # = shelf, m = mud):")
    print(env.render(env.start))
    print(
        f"\nstart={env.start}  goal={env.goal}  |S|={env.n_states()}  "
        f"|A|={N_ACTIONS}  slip_prob={env.slip_prob}  gamma={env.gamma}"
    )

    try:
        states, actions, rewards, G = simulate_episode(env, hand_crafted_policy, seed=0)
        print(f"\nOne episode under the hand-crafted policy:")
        print(f"  steps={len(actions)}  total reward={sum(rewards):.2f}  "
              f"discounted return G={G:.2f}")
        print(f"  final cell: {states[-1]} "
              f"({'goal reached' if states[-1] == env.goal else 'not at goal'})")
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    Complete TODOs T1 and T2, then re-run this script.")
        return

    try:
        for name, pol in [("random      ", random_policy), ("hand-crafted ", hand_crafted_policy)]:
            g = expected_return(env, pol, n_episodes=300, seed=0)
            print(f"expected return of {name} policy: {g:+.3f}")
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    Complete TODO T3, then re-run this script.")


if __name__ == "__main__":
    main()
