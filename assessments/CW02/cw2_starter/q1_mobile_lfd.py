"""COMP0037 2026/27 Coursework 2 — Q1 Starter Code: Mobile Manipulation Behavioral Cloning (BC).

Provides:
  - `MobilePickPlaceExpert`: 8-phase scripted FSM expert for `PandaOmron` on
    `MobilePickPlaceEnv` (drive to Table A -> hover -> descend -> grasp -> lift
    -> drive to Table B -> lower -> release).
  - `collect_demonstrations(...)`: collects successful whole-body (base + arm + gripper)
    demonstrations as `(proprio, action)` arrays.
  - `TODO(CW2-Q1b)`: `train_bc(...)` — Behavioral Cloning MLP (with optional state-noise augmentation).
  - `TODO(CW2-Q1c)`: `evaluate_bc_policy(...)` — closed-loop rollout evaluation on MobilePickPlaceEnv.
"""
from __future__ import annotations

import os
import sys

import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from mobile_pick_place_env import (  # noqa: E402
    BASE_DOCK_X,
    STATION_A_BASE,
    STATION_B_BASE,
    TARGET_PAD_POS,
    extract_proprio,
    make_mobile_pick_place_env,
)


class MobilePickPlaceExpert:
    """8-phase scripted FSM expert for PandaOmron Dual-Table Pick-and-Place."""

    PHASES = (
        "drive_to_table_a",
        "hover_cube",
        "descend_cube",
        "close_gripper",
        "lift_cube",
        "drive_to_table_b",
        "lower_to_pad",
        "release_and_retract",
    )

    def __init__(self) -> None:
        self.phase_idx = 0
        self.hold_steps = 0

    def reset(self) -> None:
        self.phase_idx = 0
        self.hold_steps = 0

    @property
    def phase(self) -> str:
        return self.PHASES[self.phase_idx]

    def act(self, obs: dict) -> np.ndarray:
        """Compute 11-D BASIC action: [arm_osc(6), torso(1), base_vel(3), gripper(1)]."""
        a = np.zeros(11, dtype=np.float64)
        base_xy = obs["robot0_base_pos"][:2]
        eef = obs["robot0_eef_pos"]
        cube = obs["cube_pos"]
        pad = TARGET_PAD_POS

        # Helper for proportional arm position control (OSC_POSE delta in base frame, base faces +x)
        def _arm_towards(target_xyz: np.ndarray, gain: float = 6.0) -> None:
            err = target_xyz - eef
            a[0:3] = np.clip(gain * err, -1.0, 1.0)

        # Helper for proportional holonomic base control (a[7]=vx, a[8]=vy)
        def _base_towards(target_xy: np.ndarray, gain_x: float = 8.0, gain_y: float = 8.0) -> float:
            err = target_xy - base_xy
            a[7] = float(np.clip(gain_x * err[0], -1.0, 1.0))
            a[8] = float(np.clip(gain_y * err[1], -1.0, 1.0))
            return float(np.linalg.norm(err))

        p = self.phase
        if p == "drive_to_table_a":
            a[10] = -1.0  # open gripper
            d_base = _base_towards(STATION_A_BASE)
            # Keep arm safely tucked at transit height while driving
            _arm_towards(np.array([base_xy[0] + 0.45, base_xy[1], 1.06]), gain=4.0)
            if d_base < 0.055:
                self.phase_idx += 1

        elif p == "hover_cube":
            a[10] = -1.0
            _base_towards(STATION_A_BASE, gain_x=2.5, gain_y=2.0)
            goal = cube + np.array([0.0, 0.0, 0.13])
            _arm_towards(goal, gain=6.0)
            if np.linalg.norm(eef - goal) < 0.03:
                self.phase_idx += 1

        elif p == "descend_cube":
            a[10] = -1.0
            _base_towards(STATION_A_BASE, gain_x=1.5, gain_y=1.5)
            goal = cube + np.array([0.0, 0.0, 0.005])
            _arm_towards(goal, gain=6.0)
            if np.linalg.norm(eef - goal) < 0.022:
                self.phase_idx += 1
                self.hold_steps = 0

        elif p == "close_gripper":
            a[10] = 1.0  # close
            _arm_towards(cube + np.array([0.0, 0.0, 0.005]), gain=3.0)
            self.hold_steps += 1
            if self.hold_steps >= 10:
                self.phase_idx += 1

        elif p == "lift_cube":
            a[10] = 1.0
            goal = np.array([cube[0], cube[1], 1.06])
            _arm_towards(goal, gain=5.5)
            if eef[2] > 1.02:
                self.phase_idx += 1

        elif p == "drive_to_table_b":
            a[10] = 1.0
            d_base = _base_towards(STATION_B_BASE)
            # Keep cube lifted cleanly above both tabletops while translating
            _arm_towards(np.array([base_xy[0] + 0.50, base_xy[1], 1.06]), gain=5.0)
            if d_base < 0.055:
                self.phase_idx += 1

        elif p == "lower_to_pad":
            a[10] = 1.0
            _base_towards(STATION_B_BASE, gain_x=2.0, gain_y=2.0)
            goal = pad + np.array([0.0, 0.0, 0.045])
            _arm_towards(goal, gain=5.5)
            if np.linalg.norm(eef - goal) < 0.03:
                self.phase_idx += 1
                self.hold_steps = 0

        elif p == "release_and_retract":
            a[10] = -1.0  # open gripper
            self.hold_steps += 1
            if self.hold_steps >= 6:
                _arm_towards(pad + np.array([0.0, 0.0, 0.18]), gain=5.0)

        return np.clip(a, -1.0, 1.0).astype(np.float32)


def collect_demonstrations(n_episodes: int = 20, seed: int = 0, horizon: int = 240) -> dict:
    """Collect successful expert demonstrations for BC and AIRL."""
    env = make_mobile_pick_place_env(use_camera_obs=False, horizon=horizon, seed=seed)
    expert = MobilePickPlaceExpert()
    all_obs, all_act, ep_lens = [], [], []
    attempts = 0
    while len(ep_lens) < n_episodes and attempts < n_episodes * 3:
        attempts += 1
        obs = env.reset()
        expert.reset()
        ep_o, ep_a = [], []
        success = False
        for _ in range(horizon):
            act = expert.act(obs)
            ep_o.append(extract_proprio(obs))
            ep_a.append(act)
            obs, _, done, _ = env.step(act)
            if env._check_success():
                success = True
                break
            if done:
                break
        if success:
            all_obs.extend(ep_o)
            all_act.extend(ep_a)
            ep_lens.append(len(ep_o))
    env.close()
    return {
        "obs": np.asarray(all_obs, dtype=np.float32),
        "actions": np.asarray(all_act, dtype=np.float32),
        "ep_lens": ep_lens,
        "success_rate": len(ep_lens) / max(1, attempts),
    }


def train_bc(
    obs_data: np.ndarray,
    act_data: np.ndarray,
    epochs: int = 120,
    noise_std: float = 0.0,
    seed: int = 0,
):
    """TODO(CW2-Q1b): Train a Behavioral Cloning MLP mapping 16-D proprio -> 11-D action.
    Normalize observations using dataset mean/std and optionally inject Gaussian input
    noise (`noise_std > 0`) during training to mitigate covariate shift.
    """
    raise NotImplementedError("TODO(CW2-Q1b): implement Behavioral Cloning (BC) policy training")


def evaluate_bc_policy(
    policy,
    n_episodes: int = 5,
    seed: int = 200,
    cube_xy_noise: float = 0.04,
    horizon: int = 240,
) -> dict:
    """TODO(CW2-Q1c): Evaluate a trained BC policy in closed loop on MobilePickPlaceEnv."""
    raise NotImplementedError("TODO(CW2-Q1c): implement closed-loop BC evaluation")


if __name__ == "__main__":
    data = collect_demonstrations(n_episodes=3, seed=0)
    print(
        f"Expert self-test: {len(data['ep_lens'])}/3 successes "
        f"(rate={data['success_rate']*100:.0f}%), mean_len={np.mean(data['ep_lens']):.1f} steps, "
        f"obs={data['obs'].shape}, act={data['actions'].shape}"
    )
