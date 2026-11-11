"""
COMP0037 — Week 02 Lab
=======================
Environments for tabular model-free RL (Q-Learning and Sarsa).

This file is COMPLETE — read it, run it, but you do not edit it here.
Your TODOs live in agents.py and are exercised through run_lab.py.

Two environments with the same interface:

  * WarehouseEnv — the Week-01 warehouse, but stochastic (slip_prob=0.2).
      Reward: -0.04 per step, -0.2 on mud, +1.0 at the goal (terminal).
  * CliffEnv     — Sutton & Barto's classic "cliff walking" gridworld.
      Reward: -1 per step, 0 on the goal-reaching step, -100 and
      reset-to-start on the cliff. Deterministic dynamics.

Interface (tabular, index-based):
    n_states, n_actions, reset(seed) -> s_idx
    step(s_idx, a) -> (s2_idx, reward, done)
"""

from __future__ import annotations

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["U", "R", "D", "L"]
ACTION_ARROWS = ["\u2191", "\u2192", "\u2193", "\u2190"]
N_ACTIONS = 4

DELTA: dict[int, tuple[int, int]] = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}
PERPENDICULAR: dict[int, tuple[int, int]] = {
    UP: (LEFT, RIGHT),
    RIGHT: (UP, DOWN),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
}


class GridWorld:
    """A generic gridworld parsed from an ASCII layout (COMPLETE).

    Layout characters:
      '.' free cell      '#' wall      'm' mud (extra cost)
      'S' start          'G' goal (terminal)     'C' cliff (penalty + reset)
    """

    LAYOUT: list[str] = []

    def __init__(self, slip_prob: float = 0.0, seed: int = 0) -> None:
        self.slip_prob = slip_prob
        self.rng = np.random.default_rng(seed)

        self.n_rows = len(self.LAYOUT)
        self.n_cols = len(self.LAYOUT[0])
        self.walls: set[tuple[int, int]] = set()
        self.mud: set[tuple[int, int]] = set()
        self.cliff: set[tuple[int, int]] = set()
        self.start: tuple[int, int] = (0, 0)
        self.goals: set[tuple[int, int]] = set()
        for r, row in enumerate(self.LAYOUT):
            for c, ch in enumerate(row):
                cell = (r, c)
                if ch == "#":
                    self.walls.add(cell)
                elif ch == "m":
                    self.mud.add(cell)
                elif ch == "C":
                    self.cliff.add(cell)
                elif ch == "S":
                    self.start = cell
                elif ch == "G":
                    self.goals.add(cell)

        self._states = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if (r, c) not in self.walls
        ]
        self.state_to_idx = {s: i for i, s in enumerate(self._states)}
        self.idx_to_state = {i: s for s, i in self.state_to_idx.items()}

    # ---- tabular interface ----
    @property
    def n_states(self) -> int:
        return len(self._states)

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.state_to_idx[self.start]

    def _is_free(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols and cell not in self.walls

    def _move(self, cell: tuple[int, int], action: int) -> tuple[int, int]:
        dr, dc = DELTA[action]
        nxt = (cell[0] + dr, cell[1] + dc)
        return nxt if self._is_free(nxt) else cell  # bump = stay

    def reward_for(self, nxt: tuple[int, int]) -> float:
        raise NotImplementedError("subclasses define rewards")

    def step(self, s_idx: int, action: int) -> tuple[int, float, bool]:
        """One (possibly stochastic) transition (COMPLETE)."""
        cell = self.idx_to_state[s_idx]
        if self.slip_prob > 0.0 and self.rng.random() < self.slip_prob:
            action = int(self.rng.choice(PERPENDICULAR[action]))
        nxt = self._move(cell, action)

        # cliff: painful reset to the start
        if nxt in self.cliff:
            return self.state_to_idx[self.start], self.reward_for(nxt), False

        r = self.reward_for(nxt)
        done = nxt in self.goals
        return self.state_to_idx[nxt], r, done

    # ---- rendering ----
    def render(self, state_idx: int | None = None) -> str:
        lines = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                cell = (r, c)
                if state_idx is not None and cell == self.idx_to_state.get(state_idx):
                    row.append("O")
                elif cell == self.start:
                    row.append("S")
                elif cell in self.goals:
                    row.append("G")
                elif cell in self.cliff:
                    row.append("C")
                elif cell in self.walls:
                    row.append("#")
                elif cell in self.mud:
                    row.append("m")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)

    def render_policy(self, Q: np.ndarray) -> str:
        """The greedy policy w.r.t. value table Q, as ASCII arrows (COMPLETE)."""
        lines = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                cell = (r, c)
                if cell in self.walls:
                    row.append("  ##  ")
                elif cell in self.goals:
                    row.append("  GG  ")
                elif cell in self.cliff:
                    row.append("  CC  ")
                else:
                    a = int(np.argmax(Q[self.state_to_idx[cell]]))
                    row.append(f"  {ACTION_ARROWS[a]}   ")
            lines.append("".join(row))
        return "\n".join(lines)


class WarehouseEnv(GridWorld):
    """Week-01 warehouse, stochastic dynamics (slip_prob = 0.2)."""

    LAYOUT = [
        "..........",
        "...##.....",
        "..........",
        "..##..mm..",
        ".....mm...",
        "S.......G.",
    ]

    def __init__(self, slip_prob: float = 0.2, seed: int = 0) -> None:
        super().__init__(slip_prob=slip_prob, seed=seed)

    def reward_for(self, nxt: tuple[int, int]) -> float:
        if nxt in self.goals:
            return 1.0
        if nxt in self.mud:
            return -0.2
        return -0.04


class CliffEnv(GridWorld):
    """Sutton & Barto cliff walking: 4 rows x 12 columns (deterministic)."""

    LAYOUT = [
        "............",
        "............",
        "............",
        "SCCCCCCCCCCG",
    ]

    def __init__(self, seed: int = 0) -> None:
        super().__init__(slip_prob=0.0, seed=seed)

    def reward_for(self, nxt: tuple[int, int]) -> float:
        if nxt in self.goals:
            return 0.0
        if nxt in self.cliff:
            return -100.0
        return -1.0


def self_test() -> None:
    """Sanity checks + demo rollouts (COMPLETE)."""
    for env_cls in (WarehouseEnv, CliffEnv):
        env = env_cls(seed=0)
        s = env.reset(seed=0)
        total, steps = 0.0, 0
        done = False
        while not done and steps < 200:
            a = int(np.random.default_rng(steps).integers(N_ACTIONS))
            s, r, done = env.step(s, a)
            total += r
            steps += 1
        print(f"{env_cls.__name__}: |S|={env.n_states}, random walk "
              f"{steps} steps, total reward {total:.1f}")
    print("\nCliffEnv layout:")
    print(CliffEnv().render())


if __name__ == "__main__":
    self_test()
