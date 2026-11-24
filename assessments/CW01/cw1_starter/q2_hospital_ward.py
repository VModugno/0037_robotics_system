"""COMP0037 2026/27 Coursework 1 — Q2 Starter Code: Hospital Ward Gridworld (8x12).

A brand-new 8x12 stochastic gridworld for MedBot:
  '.' free corridor      (reward -0.04)
  '#' wall / ward room   (impassable)
  'D' automatic door     (reward -0.50)
  'I' sterile ICU zone   (reward -5.00, high disruption cost)
  'B' biohazard spill    (reward -50.0, resets robot to Start 'S')
  'S' pharmacy start     (initial cell)
  'G' ward delivery goal (reward +10.0, terminal)
"""
from __future__ import annotations

import argparse
import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_ARROWS = ["\u2191", "\u2192", "\u2193", "\u2190"]
DELTA = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
PERPENDICULAR = {UP: (LEFT, RIGHT), RIGHT: (UP, DOWN), DOWN: (LEFT, RIGHT), LEFT: (UP, DOWN)}


class HospitalWardEnv:
    """8x12 stochastic hospital ward navigation MDP."""

    LAYOUT = [
        "...........G",
        "..##..II..#.",
        "..##..II..#.",
        "..D...II..D.",
        "......##....",
        ".##...##..#.",
        ".##..BBBB.#.",
        "S...........",
    ]

    def __init__(self, slip_prob: float = 0.2, seed: int = 0) -> None:
        self.slip_prob = slip_prob
        self.rng = np.random.default_rng(seed)
        self.n_rows = len(self.LAYOUT)
        self.n_cols = len(self.LAYOUT[0])

        self.walls: set[tuple[int, int]] = set()
        self.doors: set[tuple[int, int]] = set()
        self.icu: set[tuple[int, int]] = set()
        self.biohazard: set[tuple[int, int]] = set()
        self.start = (0, 0)
        self.goal = (0, 0)

        for r, row in enumerate(self.LAYOUT):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "D":
                    self.doors.add((r, c))
                elif ch == "I":
                    self.icu.add((r, c))
                elif ch == "B":
                    self.biohazard.add((r, c))
                elif ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)

        self.states = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if (r, c) not in self.walls
        ]
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.n_states = len(self.states)
        self.n_actions = 4
        self.pos = self.start

    def reset(self) -> int:
        self.pos = self.start
        return self.state_to_idx[self.pos]

    def _move(self, cell: tuple[int, int], action: int) -> tuple[int, int]:
        dr, dc = DELTA[action]
        nr, nc = cell[0] + dr, cell[1] + dc
        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols and (nr, nc) not in self.walls:
            return (nr, nc)
        return cell

    def step(self, action: int) -> tuple[int, float, bool]:
        u = self.rng.random()
        if u < 1.0 - self.slip_prob:
            eff = action
        elif u < 1.0 - 0.5 * self.slip_prob:
            eff = PERPENDICULAR[action][0]
        else:
            eff = PERPENDICULAR[action][1]

        nxt = self._move(self.pos, eff)
        if nxt in self.biohazard:
            self.pos = self.start
            return self.state_to_idx[self.pos], -50.0, False
        self.pos = nxt
        if nxt == self.goal:
            return self.state_to_idx[nxt], 10.0, True
        if nxt in self.icu:
            return self.state_to_idx[nxt], -5.0, False
        if nxt in self.doors:
            return self.state_to_idx[nxt], -0.5, False
        return self.state_to_idx[nxt], -0.04, False

    def render_policy(self, q_table: np.ndarray) -> str:
        lines = []
        for r in range(self.n_rows):
            row_s = []
            for c in range(self.n_cols):
                cell = (r, c)
                if cell in self.walls:
                    row_s.append(" ## ")
                elif cell == self.goal:
                    row_s.append(" GG ")
                elif cell in self.biohazard:
                    row_s.append(" BB ")
                else:
                    a = int(np.argmax(q_table[self.state_to_idx[cell]]))
                    row_s.append(f"  {ACTION_ARROWS[a]} ")
            lines.append("".join(row_s))
        return "\n".join(lines)


def train_tabular(
    env: HospitalWardEnv,
    method: str,
    episodes: int = 800,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """TODO(CW1-Q2b): Implement Q-Learning ('q_learning') and SARSA ('sarsa') on HospitalWardEnv."""
    raise NotImplementedError("TODO(CW1-Q2b): implement Q-Learning and SARSA training loop")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CW1 Q2: HospitalWardEnv self-test")
    ap.add_argument("--slip", type=float, default=0.2)
    args = ap.parse_args()
    env = HospitalWardEnv(slip_prob=args.slip, seed=0)
    print(f"HospitalWardEnv: {env.n_rows}x{env.n_cols}, |S|={env.n_states}, start={env.start}, goal={env.goal}")
    print(f"  walls={len(env.walls)}, ICU={len(env.icu)}, biohazard={len(env.biohazard)}, doors={len(env.doors)}")
    for row in env.LAYOUT:
        print("  " + row)

