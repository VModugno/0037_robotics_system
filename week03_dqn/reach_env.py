"""
COMP0037 — Week 03 Lab
=======================
A 2-link planar robot arm that must reach a target with DISCRETE actions —
the robotic task your DQN will learn, in ~150 lines with nothing hidden.

This file is COMPLETE except for TODO(T1) — the forward kinematics, i.e.
the "inner mechanics" of the arm itself. You will not fully understand the
observations your network sees until you have written it.

Interface (gym-like):
    obs = env.reset(seed=0)          # np.ndarray (6,), float32
    obs, r, done, info = env.step(3) # action index in [0, 9)

Observation: [cos t1, sin t1, cos t2, sin t2, target_x, target_y]
Actions:     9 joint-velocity commands (du, dv) in {-1, 0, 1} x {-1, 0, 1}
             applied as  theta += (du, dv) * STEP_ANGLE  (radians)

Run me (environment self-test):
    pixi run -e week03 python reach_env.py
"""

from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------- arm model
L1, L2 = 0.45, 0.35                       # link lengths (m)
JOINT_LIMITS = (2.6, 2.6)                 # |theta_i| <= 2.6 rad (hard limits)
STEP_ANGLE = 0.05                         # radians per unit action component
TARGET_EPS = 0.05                         # success radius (m)
EPISODE_LEN = 100                         # steps per episode

# 9 discrete actions: (dtheta1, dtheta2) in {-1, 0, 1}^2
ACTION_DELTAS: list[tuple[int, int]] = [
    (du, dv) for du in (-1, 0, 1) for dv in (-1, 0, 1)
]
N_ACTIONS = len(ACTION_DELTAS)  # 9


def forward_kinematics(theta1: float, theta2: float) -> tuple[float, float]:
    """Position of the arm's end-effector given joint angles (TODO T1).

    The arm: joint 1 at the origin, link 1 at absolute angle theta1,
    link 2 at absolute angle theta1 + theta2 (a *planar 2R arm*):

        x = L1 * cos(theta1) + L2 * cos(theta1 + theta2)
        y = L1 * sin(theta1) + L2 * sin(theta1 + theta2)

    Return (x, y).
    """
    raise NotImplementedError(
        "TODO(T1): implement forward_kinematics — see LAB.md."
    )


# ------------------------------------------------------------------- env
class ArmReachEnv:
    """2-link arm + random target. Reward = -distance (+1 bonus on success)."""

    state_dim = 6
    n_actions = N_ACTIONS

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)
        self.theta = np.zeros(2)
        self.target = np.zeros(2)
        self.t = 0

    # ------------------------------------------------------------- helpers
    @property
    def thetas(self) -> np.ndarray:
        return self.theta.copy()

    def _ee(self) -> np.ndarray:
        x, y = forward_kinematics(self.theta[0], self.theta[1])
        return np.array([x, y])

    def _obs(self) -> np.ndarray:
        t1, t2 = self.theta
        return np.array(
            [math.cos(t1), math.sin(t1), math.cos(t2), math.sin(t2),
             self.target[0], self.target[1]],
            dtype=np.float32,
        )

    # ----------------------------------------------------------------- api
    def reset(self, seed: int | None = None) -> np.ndarray:
        """New episode: random arm pose + random reachable target (COMPLETE)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.theta = self.rng.uniform(-1.0, 1.0, size=2)
        radius = self.rng.uniform(0.30, 0.70)      # inside reach (L1+L2=0.8)
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        self.target = radius * np.array([math.cos(angle), math.sin(angle)])
        self.t = 0
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Apply a discrete joint-velocity command (COMPLETE)."""
        du, dv = ACTION_DELTAS[action]
        self.theta = np.clip(
            self.theta + np.array([du, dv]) * STEP_ANGLE,
            -np.array(JOINT_LIMITS), np.array(JOINT_LIMITS),
        )
        self.t += 1

        dist = float(np.linalg.norm(self._ee() - self.target))
        reward = -dist
        success = dist < TARGET_EPS
        if success:
            reward += 1.0
        timeout = self.t >= EPISODE_LEN
        done = success or timeout

        info = {
            "dist": dist,
            "success": success,
            "timeout": timeout,   # True => the episode was TRUNCATED, not terminal
            "ee": self._ee(),
        }
        return self._obs(), float(reward), done, info

    # ------------------------------------------------------------ rendering
    def render(self) -> str:
        """ASCII sketch of the arm and target (COMPLETE)."""
        size = 21
        grid = [[" "] * size for _ in range(size)]

        def to_cell(x: float, y: float) -> tuple[int, int]:
            cx = int((x + 0.8) / 1.6 * (size - 1))
            cy = int((0.8 - y) / 1.6 * (size - 1))  # rows go down, y goes up
            return min(max(cy, 0), size - 1), min(max(cx, 0), size - 1)

        def draw(p_from, p_to, ch: str) -> None:
            for f in np.linspace(0.0, 1.0, 40):
                x = p_from[0] + f * (p_to[0] - p_from[0])
                y = p_from[1] + f * (p_to[1] - p_from[1])
                grid[to_cell(x, y)[0]][to_cell(x, y)[1]] = ch

        t1, t2 = self.theta
        elbow = (L1 * math.cos(t1), L1 * math.sin(t1))
        ee = self._ee()
        draw((0.0, 0.0), elbow, "1")
        draw(elbow, ee, "2")
        grid[to_cell(0, 0)[0]][to_cell(0, 0)[1]] = "B"   # base
        grid[to_cell(*ee)[0]][to_cell(*ee)[1]] = "E"      # end-effector
        grid[to_cell(*self.target)[0]][to_cell(*self.target)[1]] = "T"
        return "\n".join("".join(row) for row in grid)

    @staticmethod
    def plot_arm(theta: np.ndarray, target: np.ndarray, path=None,
                 fname: str = "arm_episode.png") -> None:
        """Save a figure of the arm, its target, and the EE path (COMPLETE)."""
        import matplotlib.pyplot as plt

        t1, t2 = theta
        elbow = (L1 * math.cos(t1), L1 * math.sin(t1))
        ee = forward_kinematics(t1, t2)
        fig, ax = plt.subplots(figsize=(5, 5))
        if path is not None and len(path) > 1:
            path = np.asarray(path)
            ax.plot(path[:, 0], path[:, 1], "-", color="0.8", lw=1.0,
                    label="EE path", zorder=1)
        ax.plot([0, elbow[0]], [0, elbow[1]], "o-", color="tab:blue", lw=3,
                label="link 1", zorder=2)
        ax.plot([elbow[0], ee[0]], [elbow[1], ee[1]], "o-", color="tab:orange",
                lw=3, label="link 2", zorder=2)
        ax.plot(*ee, "k*", ms=14, label="end-effector", zorder=3)
        ax.plot(*target, "rX", ms=12, label="target", zorder=3)
        ax.add_patch(plt.Circle((0, 0), L1 + L2, fill=False, ls=":", color="0.6"))
        ax.set_aspect("equal")
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-0.85, 0.85)
        ax.legend(fontsize=8)
        ax.set_title("2-link arm")
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"saved: {fname}")


# --------------------------------------------------------------- self-test
def self_test() -> None:
    """Checks your T1, then exercises the environment (runs automatically)."""
    # 1) forward kinematics sanity — geometric ground truth
    x, y = forward_kinematics(0.0, 0.0)
    assert abs(x - (L1 + L2)) < 1e-9 and abs(y) < 1e-9, \
        "FK(0, 0) should be the fully stretched arm at (L1+L2, 0)"
    x, y = forward_kinematics(math.pi / 2, 0.0)
    assert abs(x) < 1e-9 and abs(y - (L1 + L2)) < 1e-9, \
        "FK(pi/2, 0) should point the whole arm straight up at (0, L1+L2)"
    x, y = forward_kinematics(0.0, math.pi)
    assert abs(x - (L1 - L2)) < 1e-9 and abs(y) < 1e-9, \
        "FK(0, pi) should fold link 2 straight back"
    print("forward kinematics checks passed.")

    # 2) roll out a random policy
    env = ArmReachEnv()
    wins, dists = 0, []
    for ep in range(10):
        obs = env.reset(seed=ep)
        assert obs.shape == (6,) and obs.dtype == np.float32
        done, steps = False, 0
        while not done:
            obs, r, done, info = env.step(int(np.random.default_rng(steps).integers(9)))
            steps += 1
        wins += info["success"]
        dists.append(info["dist"])
    print(f"random policy over 10 episodes: successes={wins}/10, "
          f"mean final dist={np.mean(dists):.3f} m")
    print("\nOne random start pose (B=base, 1/2=links, E=end-effector, T=target):")
    env.reset(seed=0)
    print(env.render())


if __name__ == "__main__":
    try:
        self_test()
    except NotImplementedError as e:
        print(f"[!] {e}")
        print("    Complete TODO(T1) in reach_env.py, then re-run this file.")
