"""COMP0037 2026/27 Coursework 2 — Q4 Starter Code: Whole-Body CBF Safety Shield & Red-Teaming.

Integrates the 4-tier safe mobile-manipulation stack on `MobilePickPlaceEnv`:
  Tier 1: High-level LLM / Mock Task Planner (`MobileLLMPlanner`)
  Tier 2: Closed-loop Mobile-Manipulation Skills (`MobileSkillLibrary`)
  Tier 3: Operational-Space & Mobile-Base Whole-Body CBF Safety Shield (`WholeBodyCBFShield`)
  Tier 4: `PandaOmron` Robot in Robosuite 1.5.2

Student TODOs:
  - `TODO(CW2-Q4b)`: `WholeBodyCBFShield.filter_action(obs, u_nom)` enforcing:
      1. Mobile base table-collision barrier:  h_x(p) = AISLE_X_MAX - x_base >= 0
      2. Mobile base aisle lateral bounds:     AISLE_Y_MIN <= y_base <= AISLE_Y_MAX
      3. End-effector table-impact barrier:    h_z(e) = z_ee - EE_Z_MIN >= 0
      4. Velocity & action-norm clamping + structured safety audit log.
  - `TODO(CW2-Q4c)`: `run_red_team_suite(env_factory, shield)` evaluating 3 scenarios:
      (i) Nominal transfer, (ii) Adversarial commands ("ram table B" / "slam cube"),
      (iii) Noisy perception (injected Gaussian pose error on cube & base odometry).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from mobile_pick_place_env import (  # noqa: E402
    AISLE_X_MAX,
    AISLE_X_MIN,
    AISLE_Y_MAX,
    AISLE_Y_MIN,
    EE_Z_MIN,
    make_mobile_pick_place_env,
)


@dataclass
class SafetyAuditEntry:
    step: int
    reason: str
    u_nom_norm: float
    u_safe_norm: float
    base_x: float
    eef_z: float


class WholeBodyCBFShield:
    """Coupled mobile-base + manipulator Control Barrier Function safety filter."""

    def __init__(
        self,
        alpha_base: float = 0.8,
        alpha_ee: float = 0.8,
        max_base_vel: float = 0.85,
        max_arm_cmd: float = 0.90,
        dt: float = 0.05,
    ) -> None:
        self.alpha_base = alpha_base
        self.alpha_ee = alpha_ee
        self.max_base_vel = max_base_vel
        self.max_arm_cmd = max_arm_cmd
        self.dt = dt
        self.step_count = 0
        self.interventions = 0
        self.audit_log: list[SafetyAuditEntry] = []

    def reset(self) -> None:
        self.step_count = 0
        self.interventions = 0
        self.audit_log.clear()

    def filter_action(self, obs: dict, u_nom: np.ndarray) -> np.ndarray:
        """TODO(CW2-Q4b): Project nominal 11-D action `u_nom` to satisfy base + EE CBF barriers."""
        raise NotImplementedError("TODO(CW2-Q4b): implement WholeBodyCBFShield.filter_action")


def run_red_team_suite(shield_enabled: bool = True, seed: int = 0) -> dict:
    """TODO(CW2-Q4c): Execute nominal, adversarial, and noisy-perception red-team scenarios."""
    raise NotImplementedError("TODO(CW2-Q4c): implement 3-scenario red-teaming suite")


if __name__ == "__main__":
    env = make_mobile_pick_place_env(use_camera_obs=False, seed=0)
    obs = env.reset()
    print(
        f"WholeBodyCBFShield ready: aisle_x in [{AISLE_X_MIN}, {AISLE_X_MAX}], "
        f"aisle_y in [{AISLE_Y_MIN}, {AISLE_Y_MAX}], ee_z >= {EE_Z_MIN}"
    )
    print(f"  initial base_x={obs['robot0_base_pos'][0]:.3f}, eef_z={obs['robot0_eef_pos'][2]:.3f}")
    env.close()

