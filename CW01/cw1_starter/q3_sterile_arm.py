"""COMP0037 2026/27 Coursework 1 — Q3 Starter Code: Sterile Cabinet 2-DoF Arm.

A 2-DoF planar sample-transfer arm operating inside a hospital biosafety cabinet
with BOTH:
  1. Hard joint-angle limits: q_i in [Q_MIN, Q_MAX] = [-1.6, 1.6] rad
  2. An internal circular centrifuge keep-out disk:
     ||FK(q) - OBS_CENTER||_2 >= OBS_RADIUS = 0.16 m

Supports discrete 9-action mode (for DQN / Double DQN) and continuous velocity
mode (for Unshielded / Soft-Penalty / Dual-Barrier CBF-Shielded A2C).
"""
from __future__ import annotations

import math
import numpy as np

L1, L2 = 0.55, 0.45
Q_MIN = np.array([-1.6, -1.6], dtype=np.float64)
Q_MAX = np.array([1.6, 1.6], dtype=np.float64)
OBS_CENTER = np.array([0.45, 0.50], dtype=np.float64)  # Fragile benchtop centrifuge
OBS_RADIUS = 0.16
DT = 0.10
MAX_VEL = 1.2
DISCRETE_DELTAS = np.array(
    [(d1, d2) for d1 in (-0.12, 0.0, 0.12) for d2 in (-0.12, 0.0, 0.12)],
    dtype=np.float64,
)


def forward_kinematics(theta1: float, theta2: float) -> np.ndarray:
    x = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
    y = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
    return np.array([x, y], dtype=np.float64)


def jacobian(theta1: float, theta2: float) -> np.ndarray:
    s1, c1 = math.sin(theta1), math.cos(theta1)
    s12, c12 = math.sin(theta1 + theta2), math.cos(theta1 + theta2)
    return np.array(
        [[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]],
        dtype=np.float64,
    )


class SterileArmEnv:
    """2-DoF planar arm with hard joint limits and circular centrifuge keep-out disk."""

    def __init__(self, mode: str = "continuous", max_steps: int = 120, seed: int = 0) -> None:
        assert mode in ("discrete", "continuous")
        self.mode = mode
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros(2, dtype=np.float64)
        self.target = np.array([0.15, 0.78], dtype=np.float64)
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Sample starting configuration in the lower-right staging zone (below centrifuge)
        while True:
            q0 = self.rng.uniform([-0.75, -0.55], [0.10, 0.55])
            ee0 = forward_kinematics(*q0)
            if np.linalg.norm(ee0 - OBS_CENTER) > OBS_RADIUS + 0.10:
                self.q = q0
                break
        # Sample reachable target vial on the far side (upper-left) of the centrifuge
        while True:
            tq = self.rng.uniform([0.85, -0.55], [1.45, 0.55])
            cand = forward_kinematics(*tq)
            if (
                np.linalg.norm(cand - OBS_CENTER) > OBS_RADIUS + 0.08
                and np.linalg.norm(cand - ee0) > 0.45
            ):
                self.target = cand
                break
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        ee = forward_kinematics(*self.q)
        return np.concatenate([self.q, ee, self.target]).astype(np.float32)

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        prev_dist = float(np.linalg.norm(forward_kinematics(*self.q) - self.target))
        if self.mode == "discrete":
            dq = DISCRETE_DELTAS[int(action)]
            q_next = self.q + dq
        else:
            vel = np.clip(np.asarray(action, dtype=np.float64), -MAX_VEL, MAX_VEL)
            q_next = self.q + vel * DT

        limit_hit = bool(np.any(q_next < Q_MIN) or np.any(q_next > Q_MAX))
        self.q = np.clip(q_next, Q_MIN, Q_MAX)
        ee = forward_kinematics(*self.q)
        obs_margin = float(np.linalg.norm(ee - OBS_CENTER) - OBS_RADIUS)
        obs_hit = obs_margin < 0.0

        dist = float(np.linalg.norm(ee - self.target))
        self.steps += 1
        success = dist < 0.06 and not obs_hit
        done = success or (self.steps >= self.max_steps)
        act_pen = 0.01 * float(np.sum(np.square(action))) if self.mode == "continuous" else 0.0
        reward = 2.0 * (prev_dist - dist) - 0.4 * dist - act_pen
        if success:
            reward += 5.0
        info = {
            "dist": dist,
            "success": success,
            "limit_hit": limit_hit,
            "obs_hit": obs_hit,
            "obs_margin": obs_margin,
            "truncated": (self.steps >= self.max_steps) and not success,
        }
        return self._obs(), float(reward), done, info


def cbf_safety_shield(q: np.ndarray, u_nom: np.ndarray, alpha_obs: float = 0.5) -> np.ndarray:
    """TODO(CW1-Q3d): Project nominal velocity u_nom to satisfy BOTH:
      1. Hard joint limits: Q_MIN <= q + u * DT <= Q_MAX
      2. Circular centrifuge barrier: h_obs(q) = ||FK(q) - OBS_CENTER||^2 - OBS_RADIUS^2 >= 0
         with discrete-time CBF condition grad_h(q)^T u * DT + alpha_obs * h_obs(q) >= 0,
         where grad_h(q) = 2 * J(q)^T (FK(q) - OBS_CENTER).

    Hint: Closed-form half-space projection (no QP solver needed):
      ee = forward_kinematics(*q)
      h = ||ee - OBS_CENTER||^2 - r_safe^2
      g = 2 * jacobian(*q).T @ (ee - OBS_CENTER) * DT
      if g @ u + alpha_obs * h < 0:
          u = u - ((g @ u + alpha_obs * h) / (g @ g + 1e-8)) * g
    interleaved with joint-limit velocity clamping.
    """
    raise NotImplementedError("TODO(CW1-Q3d): implement dual-barrier CBF safety shield")


if __name__ == "__main__":
    env = SterileArmEnv(mode="continuous", seed=0)
    obs = env.reset(seed=0)
    print(f"SterileArmEnv ready: obs_dim={obs.shape[0]}, start_ee={obs[2:4].round(3)}, target={obs[4:6].round(3)}")
    print(f"  joint limits=[{Q_MIN[0]}, {Q_MAX[0]}] rad, centrifuge at {OBS_CENTER} (r={OBS_RADIUS} m)")
