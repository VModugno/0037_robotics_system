"""COMP0037 2026/27 Coursework 2 — Q2 Starter Code: LLM Task Planning & Typed Mobile Skill API.

Exposes `PandaOmron` on `MobilePickPlaceEnv` through a typed Mobile-Manipulation Skill API:
  - `drive_to_station(station="A"|"B")`
  - `stow_arm()`
  - `pick_cube()`
  - `place_cube()`
  - `verify_grasp()`

Student TODOs:
  - `TODO(CW2-Q2b)`: `validate_plan(plan, world_state)` — schema & precondition validator
    (rejects unknown skills/arguments, picking when not docked at Table A, driving to
    Table B while the arm is below safe transit height, or placing without holding the cube).
  - `TODO(CW2-Q2c)`: `MobileLLMPlanner.plan(instruction, world_state)` — structured JSON
    task planner (Mock / Ollama / OpenAI) + closed-loop failure recovery loop.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from mobile_pick_place_env import (  # noqa: E402
    STATION_A_BASE,
    STATION_B_BASE,
    TARGET_PAD_POS,
    make_mobile_pick_place_env,
)

ALLOWED_SKILLS = {
    "drive_to_station": {"station": ("A", "B")},
    "stow_arm": {},
    "pick_cube": {},
    "place_cube": {},
    "verify_grasp": {},
}


@dataclass
class SkillResult:
    skill: str
    success: bool
    steps: int
    message: str


class MobileSkillLibrary:
    """Closed-loop proportional skills controlling PandaOmron (base + arm + gripper)."""

    def __init__(self, env, action_filter=None, record_frames: bool = False) -> None:
        self.env = env
        self.action_filter = action_filter  # Optional WholeBodyCBFShield callback (used in Q4)
        self.record_frames = record_frames
        self.frames: list[np.ndarray] = []
        self.obs = None

    def reset_env(self) -> dict:
        self.frames.clear()
        self.obs = self.env.reset()
        self._maybe_capture()
        return self.obs

    def _maybe_capture(self) -> None:
        if self.record_frames and "agentview_image" in self.obs:
            self.frames.append(np.flipud(self.obs["agentview_image"]).copy())

    def _step(self, raw_action: np.ndarray) -> dict:
        act = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        if self.action_filter is not None:
            act = self.action_filter(self.obs, act)
        self.obs, _, _, _ = self.env.step(act)
        self._maybe_capture()
        return self.obs

    def get_world_state(self) -> dict:
        base_xy = self.obs["robot0_base_pos"][:2]
        eef = self.obs["robot0_eef_pos"]
        cube = self.obs["cube_pos"]
        d_a = float(np.linalg.norm(base_xy - STATION_A_BASE))
        d_b = float(np.linalg.norm(base_xy - STATION_B_BASE))
        station = "A" if d_a < 0.10 else ("B" if d_b < 0.10 else "aisle")
        return {
            "station": station,
            "base_xy": base_xy.round(3).tolist(),
            "eef_z": round(float(eef[2]), 3),
            "arm_stowed": bool(eef[2] >= 1.00),
            "holding_cube": bool(self.env.is_grasping_cube() and cube[2] > 0.86),
            "task_success": bool(self.env._check_success()),
        }

    def stow_arm(self, max_steps: int = 30) -> SkillResult:
        hold_grip = 1.0 if self.env.is_grasping_cube() else -1.0
        for t in range(1, max_steps + 1):
            base_xy = self.obs["robot0_base_pos"][:2]
            eef = self.obs["robot0_eef_pos"]
            goal = np.array([base_xy[0] + 0.50, base_xy[1], 1.06])
            a = np.zeros(11, dtype=np.float32)
            a[0:3] = np.clip(5.5 * (goal - eef), -1.0, 1.0)
            a[10] = hold_grip
            self._step(a)
            if eef[2] >= 1.02:
                return SkillResult("stow_arm", True, t, "Arm raised to safe transit height.")
        return SkillResult("stow_arm", False, max_steps, "Timed out raising arm.")

    def drive_to_station(self, station: str = "A", max_steps: int = 75) -> SkillResult:
        target_xy = STATION_A_BASE if station == "A" else STATION_B_BASE
        hold_grip = 1.0 if self.env.is_grasping_cube() else -1.0
        for t in range(1, max_steps + 1):
            base_xy = self.obs["robot0_base_pos"][:2]
            eef = self.obs["robot0_eef_pos"]
            err = target_xy - base_xy
            a = np.zeros(11, dtype=np.float32)
            a[7] = float(np.clip(8.0 * err[0], -1.0, 1.0))
            a[8] = float(np.clip(8.0 * err[1], -1.0, 1.0))
            # Maintain safe transit arm posture relative to base
            goal_eef = np.array([base_xy[0] + 0.50, base_xy[1], 1.06])
            a[0:3] = np.clip(5.0 * (goal_eef - eef), -1.0, 1.0)
            a[10] = hold_grip
            self._step(a)
            if float(np.linalg.norm(err)) < 0.055:
                return SkillResult("drive_to_station", True, t, f"Docked at station {station}.")
        return SkillResult("drive_to_station", False, max_steps, f"Failed to reach station {station}.")

    def pick_cube(self, max_steps: int = 85) -> SkillResult:
        steps = 0
        # 1. Hover above cube
        for _ in range(30):
            steps += 1
            cube = self.obs["cube_pos"]
            eef = self.obs["robot0_eef_pos"]
            goal = cube + np.array([0.0, 0.0, 0.13])
            a = np.zeros(11, dtype=np.float32)
            a[0:3] = np.clip(6.0 * (goal - eef), -1.0, 1.0)
            a[10] = -1.0
            self._step(a)
            if np.linalg.norm(eef - goal) < 0.03:
                break
        # 2. Descend to cube
        for _ in range(25):
            steps += 1
            cube = self.obs["cube_pos"]
            eef = self.obs["robot0_eef_pos"]
            goal = cube + np.array([0.0, 0.0, 0.005])
            a = np.zeros(11, dtype=np.float32)
            a[0:3] = np.clip(6.0 * (goal - eef), -1.0, 1.0)
            a[10] = -1.0
            self._step(a)
            if np.linalg.norm(eef - goal) < 0.022:
                break
        # 3. Close gripper
        for _ in range(10):
            steps += 1
            a = np.zeros(11, dtype=np.float32)
            a[10] = 1.0
            self._step(a)
        # 4. Lift clear of Table A
        for _ in range(20):
            steps += 1
            cube = self.obs["cube_pos"]
            eef = self.obs["robot0_eef_pos"]
            goal = np.array([cube[0], cube[1], 1.06])
            a = np.zeros(11, dtype=np.float32)
            a[0:3] = np.clip(5.5 * (goal - eef), -1.0, 1.0)
            a[10] = 1.0
            self._step(a)
            if eef[2] > 1.02:
                break
        ok = bool(self.env.is_grasping_cube() and self.obs["cube_pos"][2] > 0.88)
        return SkillResult("pick_cube", ok, steps, "Cube grasped and lifted." if ok else "Grasp failed.")

    def place_cube(self, max_steps: int = 55) -> SkillResult:
        steps = 0
        # 1. Lower above Table B green target pad
        for _ in range(30):
            steps += 1
            eef = self.obs["robot0_eef_pos"]
            goal = TARGET_PAD_POS + np.array([0.0, 0.0, 0.045])
            a = np.zeros(11, dtype=np.float32)
            a[0:3] = np.clip(5.5 * (goal - eef), -1.0, 1.0)
            a[10] = 1.0
            self._step(a)
            if np.linalg.norm(eef - goal) < 0.03:
                break
        # 2. Release gripper and retract
        for t in range(18):
            steps += 1
            eef = self.obs["robot0_eef_pos"]
            a = np.zeros(11, dtype=np.float32)
            a[10] = -1.0
            if t >= 6:
                goal = TARGET_PAD_POS + np.array([0.0, 0.0, 0.18])
                a[0:3] = np.clip(5.0 * (goal - eef), -1.0, 1.0)
            self._step(a)
        ok = bool(self.env._check_success())
        return SkillResult("place_cube", ok, steps, "Cube placed on Table B." if ok else "Placement check failed.")

    def verify_grasp(self) -> SkillResult:
        ok = bool(self.env.is_grasping_cube() and self.obs["cube_pos"][2] > 0.86)
        return SkillResult("verify_grasp", ok, 0, "Grasp verified." if ok else "Cube not held!")

    def execute(self, skill_name: str, args: dict | None = None) -> SkillResult:
        args = args or {}
        if skill_name == "drive_to_station":
            return self.drive_to_station(station=str(args.get("station", "A")))
        if skill_name == "stow_arm":
            return self.stow_arm()
        if skill_name == "pick_cube":
            return self.pick_cube()
        if skill_name == "place_cube":
            return self.place_cube()
        if skill_name == "verify_grasp":
            return self.verify_grasp()
        return SkillResult(skill_name, False, 0, f"Unknown skill: {skill_name}")


def validate_plan(plan: list[dict], initial_state: dict) -> tuple[bool, list[str]]:
    """TODO(CW2-Q2b): Validate JSON skill schema AND symbolic preconditions.

    Must reject:
      1. Unknown skill names or invalid argument values (e.g., station not in {'A', 'B'}).
      2. Calling `pick_cube` when the robot is not docked at station 'A'.
      3. Calling `drive_to_station` while the arm is not stowed (arm_stowed == False).
      4. Calling `place_cube` when not docked at station 'B' or not holding the cube.
    Returns `(is_valid, list_of_error_messages)`.
    """
    raise NotImplementedError("TODO(CW2-Q2b): implement plan schema & precondition validator")


class MobileLLMPlanner:
    """TODO(CW2-Q2c): Structured JSON LLM planner (Mock / Ollama / OpenAI) + recovery."""

    def __init__(self, backend: str = "mock") -> None:
        self.backend = backend

    def plan(self, instruction: str, world_state: dict) -> list[dict]:
        """Return a JSON list of skill dicts: [{'skill': str, 'args': dict}, ...]."""
        raise NotImplementedError("TODO(CW2-Q2c): implement MobileLLMPlanner.plan")


if __name__ == "__main__":
    env = make_mobile_pick_place_env(use_camera_obs=False, seed=0)
    lib = MobileSkillLibrary(env)
    lib.reset_env()
    print("Initial world state:", lib.get_world_state())
    env.close()
