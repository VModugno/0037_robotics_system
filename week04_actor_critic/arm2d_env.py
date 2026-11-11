"""
COMP0037 — Week 04 Lab
=======================
The SAME 2-link planar arm as Week 03, but now with CONTINUOUS actions —
the environment for your Actor-Critic agent.

This file is COMPLETE (no TODOs): the only change from Week 03 is the
action space. Read step() closely: hard joint limits act exactly like the
safety constraints you will study in the lecture (CBFs).

Interface:
    obs = env.reset(seed=0)                    # (6,) float32, same as week 03
    obs, r, done, info = env.step(np.array([0.3, -0.8]))  # action in [-1, 1]^2

Action semantics:  a in [-1,1]^2  ->  joint velocities  omega = a * OMEGA_MAX,
integrated for DT seconds. Hard joint limits are enforced by clamping; every
clamp is counted as a *limit violation* (your safety metric).

Run me (self-test):
    pixi run -e week04 python arm2d_env.py
"""

from __future__ import annotations

import math

import numpy as np

L1, L2 = 0.45, 0.35
JOINT_LIMITS = (2.6, 2.6)     # |theta_i| <= 2.6 rad
OMEGA_MAX = 2.5               # rad/s at full action
DT = 0.05                     # seconds per step
EPISODE_LEN = 150             # steps
TARGET_EPS = 0.05             # success radius (m)


def forward_kinematics(theta1: float, theta2: float) -> tuple[float, float]:
    """End-effector position (same as Week 03 — now provided, you proved it)."""
    x = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
    y = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
    return x, y


class Arm2DEnv:
    state_dim = 6
    action_dim = 2

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)
        self.theta = np.zeros(2)
        self.target = np.zeros(2)
        self.t = 0
        self.limit_hits = 0   # total hard-limit clamps this episode

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

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.theta = self.rng.uniform(-1.0, 1.0, size=2)
        radius = self.rng.uniform(0.30, 0.70)
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        self.target = radius * np.array([math.cos(angle), math.sin(angle)])
        self.t = 0
        self.limit_hits = 0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """One control step: command velocities, integrate, clamp limits (COMPLETE)."""
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        omega = a * OMEGA_MAX
        new_theta = self.theta + omega * DT

        limit_hit = False
        for i in range(2):
            if abs(new_theta[i]) > JOINT_LIMITS[i]:
                new_theta[i] = np.clip(new_theta[i], -JOINT_LIMITS[i],
                                       JOINT_LIMITS[i])
                limit_hit = True
        if limit_hit:
            self.limit_hits += 1
        self.theta = new_theta
        self.t += 1

        dist = float(np.linalg.norm(self._ee() - self.target))
        success = dist < TARGET_EPS
        timeout = self.t >= EPISODE_LEN
        done = success or timeout

        # reward: get close, reach the target, do not waste energy
        reward = -dist - 0.01 * float(np.sum(omega**2))
        if success:
            reward += 1.0

        info = {
            "dist": dist,
            "success": success,
            "timeout": timeout,
            "limit_hit": limit_hit,
            "ee": self._ee(),
        }
        return self._obs(), float(reward), done, info

    def render(self) -> str:
        """ASCII sketch (COMPLETE)."""
        size = 21
        grid = [[" "] * size for _ in range(size)]

        def to_cell(x: float, y: float) -> tuple[int, int]:
            cx = int((x + 0.8) / 1.6 * (size - 1))
            cy = int((0.8 - y) / 1.6 * (size - 1))
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
        grid[to_cell(0, 0)[0]][to_cell(0, 0)[1]] = "B"
        grid[to_cell(*ee)[0]][to_cell(*ee)[1]] = "E"
        grid[to_cell(*self.target)[0]][to_cell(*self.target)[1]] = "T"
        return "\n".join("".join(row) for row in grid)


def self_test() -> None:
    env = Arm2DEnv()
    rng = np.random.default_rng(0)
    wins, hits = 0, 0
    for ep in range(10):
        obs = env.reset(seed=ep)
        assert obs.shape == (6,) and obs.dtype == np.float32
        done = False
        while not done:
            obs, r, done, info = env.step(rng.uniform(-1, 1, 2))
        wins += info["success"]
        hits += env.limit_hits
    print(f"random uniform policy: successes={wins}/10, "
          f"joint-limit clamps={hits}")
    env.reset(seed=0)
    # push into a joint limit deliberately
    env.theta = np.array([2.6, 0.0])
    _, _, _, info = env.step(np.array([1.0, 0.0]))
    assert info["limit_hit"], "full action at the limit must clamp"
    print("joint-limit clamping verified.")
    print("\nOne start pose (B=base, 1/2=links, E=end-effector, T=target):")
    print(env.render())


if __name__ == "__main__":
    self_test()
