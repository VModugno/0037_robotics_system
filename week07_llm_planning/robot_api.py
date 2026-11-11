"""Week 07 - LLMs in robotics, part 1: the robot skill API.

The LLM never speaks "robot". It emits a *plan*: a JSON list of calls to a
small, typed, safety-checkable set of skills. This file is that skill set --
the bridge between language and control. The pattern is the same one used
throughout the field (SayCan, Code-as-Policies, inner monologue, RT-X skill
hierarchies): LLM = decomposition, skill API = grounding.

Skills (all built on the same OSC_POSE P-controller you wrote in week 06):

    reset()                   start a new episode, return the fresh state
    state()                   {"eef_pos": [x,y,z], "cube_pos": [x,y,z],
                              "gripper_open": bool, "success": bool}
    move_to(x, y, z)          drive the end-effector to a world position
    move_above(target, dz)    hover dz metres above a named target ("cube")
    close_gripper() / open_gripper()
    lift(height)              raise the gripper `height` metres
    wait(steps)               hold position (the controller settles)

Skill execution details you should notice (and will report on):

  * every skill is a *closed-loop* controller, not a trajectory replay:
    it re-targets the live cube position each step, exactly like the week 06
    expert -- the LLM only picks the goal;
  * move_to is bounded: it refuses targets outside the arm's reachable
    workspace envelope (WORKSPACE_LIMITS), returning an error instead of
    crashing the arm into the table -- the first, simplest form of a safety
    filter, and the seed of week 09's CBF layer;
  * skills fail loudly: they return ("ok", info) or ("error", reason), and
    the plan executor (llm_planner.py, T4) decides what to do about it.

Usage:
    pixi run --manifest-path ../pixi.toml -e week07 python robot_api.py --demo
"""

import argparse
import os

import numpy as np

# --------------------------------------------------------------------------- #
# robosuite import with Windows-friendly preparation (same as week 06)
# --------------------------------------------------------------------------- #

if os.name == "nt":
    os.makedirs("C:\\tmp", exist_ok=True)

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


class RobotSkillAPI:
    """A small typed skill set over the robosuite Lift task.

    The skill API is the *contract* the LLM must satisfy. Its signature
    doubles as the tool schema given to the model in llm_planner.py.
    """

    # conservative reachable-box around the table (world frame, metres)
    WORKSPACE_LIMITS = {
        "x": (-0.25, 0.25),
        "y": (-0.35, 0.15),
        "z": (0.80, 1.15),
    }

    GRIP_OPEN = -1.0
    GRIP_CLOSE = 1.0
    MAX_DELTA = 0.05          # metres per control step at full OSC command
    TOL = 0.02                # metres: position skill is done when inside
    SETTLE_STEPS = 6          # extra steps to let the arm settle on arrival
    GRIP_STEPS = 30

    def __init__(self, horizon=300, control_freq=20, seed=None):
        cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
        self.env = suite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=cfg,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=True,
            lite_physics=True,
        )
        self.horizon = horizon
        self.obs = None
        if seed is not None:
            np.random.seed(seed)
        self.reset(seed=None)

    # ------------------------------------------------------------------ #
    # environment plumbing
    # ------------------------------------------------------------------ #

    def reset(self, seed=None):
        """Start a new episode. seed re-seeds robosuite's placement RNG."""
        if seed is not None:
            np.random.seed(seed)
        self.obs = self.env.reset()
        return self.state()

    def state(self):
        """A compact, LLM-readable snapshot of the world."""
        cube = np.asarray(self.obs["object-state"], dtype=np.float64)[:3]
        return {
            "eef_pos": [round(float(v), 3) for v in self.obs["robot0_eef_pos"]],
            "cube_pos": [round(float(v), 3) for v in cube],
            "gripper_open": bool(self.obs["robot0_gripper_qpos"][0] < -0.02),
            "success": bool(self.env._check_success()),
        }

    def close(self):
        self.env.close()

    # ------------------------------------------------------------------ #
    # low-level controller (the week 06 P-controller, factored out)
    # ------------------------------------------------------------------ #

    def _drive_to(self, target, grip, tol=None, max_steps=60):
        """Closed-loop P-controller toward `target` with gripper `grip`.

        Returns (reached: bool, steps: int). Targets outside the workspace
        box are refused up front (the simplest possible safety check).
        """
        target = np.asarray(target, dtype=np.float64)
        err, reason = self._check_workspace(target)
        if err:
            return False, reason
        tol = self.TOL if tol is None else tol
        for t in range(max_steps):
            eef = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64)
            action = np.zeros(7)
            action[:3] = np.clip((target - eef) / self.MAX_DELTA, -1.0, 1.0)
            action[6] = grip
            self.obs, *_ = self.env.step(action)
            if np.linalg.norm(target - eef) < tol:
                for _ in range(self.SETTLE_STEPS):
                    hold = np.zeros(7)
                    hold[6] = grip
                    self.obs, *_ = self.env.step(hold)
                return True, t + 1
        return False, "timeout"

    def _check_workspace(self, target):
        """Refuse targets outside the conservative reachable box."""
        for axis, (lo, hi) in self.WORKSPACE_LIMITS.items():
            v = target["xyz".index(axis)]
            if not (lo - 1e-9 <= v <= hi + 1e-9):
                return True, (
                    f"target {axis}={v:.3f} outside workspace "
                    f"[{lo:.2f}, {hi:.2f}]"
                )
        return False, ""

    # ------------------------------------------------------------------ #
    # the skill set (the LLM's "tools")
    # ------------------------------------------------------------------ #

    def skill_move_to(self, x, y, z):
        ok, info = self._drive_to((x, y, z), self.GRIP_OPEN)
        return ("ok", {"steps": info}) if ok else ("error", info)

    def skill_move_above(self, target="cube", dz=0.12):
        """Hover dz metres above a named object. Tracks the object live."""
        if target != "cube":
            return "error", f"unknown target {target!r} (only 'cube')"
        cube = np.asarray(self.obs["object-state"], dtype=np.float64)[:3]
        ok, info = self._drive_to(cube + (0.0, 0.0, dz), self.GRIP_OPEN)
        return ("ok", {"steps": info}) if ok else ("error", info)

    def skill_close_gripper(self):
        for _ in range(self.GRIP_STEPS):
            a = np.zeros(7)
            a[6] = self.GRIP_CLOSE
            self.obs, *_ = self.env.step(a)
        return "ok", {"gripper": "closed"}

    def skill_open_gripper(self):
        for _ in range(self.GRIP_STEPS):
            a = np.zeros(7)
            a[6] = self.GRIP_OPEN
            self.obs, *_ = self.env.step(a)
        return "ok", {"gripper": "open"}

    def skill_lift(self, height=0.15):
        eef = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64)
        ok, info = self._drive_to(eef + (0.0, 0.0, height), self.GRIP_CLOSE)
        return ("ok", {"steps": info}) if ok else ("error", info)

    def skill_wait(self, steps=5):
        for _ in range(max(1, int(steps))):
            hold = np.zeros(7)
            hold[6] = self.GRIP_CLOSE
            self.obs, *_ = self.env.step(hold)
        return "ok", {"waited": int(steps)}

    def skill_descend_to_cube(self, dz=0.0):
        """Descend the open gripper to dz above the cube centre (grasp prep)."""
        cube = np.asarray(self.obs["object-state"], dtype=np.float64)[:3]
        ok, info = self._drive_to(cube + (0.0, 0.0, dz), self.GRIP_OPEN)
        return ("ok", {"steps": info}) if ok else ("error", info)


# --------------------------------------------------------------------------- #
# the registry: the machine-readable "tool schema" for the LLM
# --------------------------------------------------------------------------- #

SKILLS = {
    "move_to": {
        "doc": "drive the end-effector to a world (x, y, z) position, gripper open",
        "args": {"x": "float", "y": "float", "z": "float"},
        "method": RobotSkillAPI.skill_move_to,
    },
    "move_above": {
        "doc": "hover dz metres above a named object ('cube')",
        "args": {"target": "str", "dz": "float"},
        "method": RobotSkillAPI.skill_move_above,
    },
    "descend_to_cube": {
        "doc": "descend the open gripper to dz above the cube centre",
        "args": {"dz": "float"},
        "method": RobotSkillAPI.skill_descend_to_cube,
    },
    "close_gripper": {
        "doc": "close the gripper (grasp)",
        "args": {},
        "method": RobotSkillAPI.skill_close_gripper,
    },
    "open_gripper": {
        "doc": "open the gripper (release)",
        "args": {},
        "method": RobotSkillAPI.skill_open_gripper,
    },
    "lift": {
        "doc": "raise the gripper `height` metres (grasp first!)",
        "args": {"height": "float"},
        "method": RobotSkillAPI.skill_lift,
    },
    "wait": {
        "doc": "hold position for `steps` control steps",
        "args": {"steps": "int"},
        "method": RobotSkillAPI.skill_wait,
    },
}


def call_skill(api, name, args):
    """Dispatch one skill call by name. Unknown names are an error, not a crash."""
    if name not in SKILLS:
        return "error", f"unknown skill {name!r} (known: {sorted(SKILLS)})"
    method = getattr(api, SKILLS[name]["method"].__name__)
    return method(**args)


def demo():
    """A plan written by hand -- the same plan the mock LLM will emit."""
    api = RobotSkillAPI(seed=0)
    print("initial state:", api.state())
    plan = [
        ("move_above", {"target": "cube", "dz": 0.12}),
        ("descend_to_cube", {"dz": 0.0}),
        ("close_gripper", {}),
        ("lift", {"height": 0.15}),
        ("wait", {"steps": 3}),
    ]
    for name, args in plan:
        status, info = call_skill(api, name, args)
        print(f"{name:18s} -> {status} {info}")
        if status == "error":
            break
    print("final state:", api.state())
    api.close()


def main():
    parser = argparse.ArgumentParser(description="Week 07 robot skill API")
    parser.add_argument("--demo", action="store_true", help="run the hand-written plan")
    parser.add_argument("--list-skills", action="store_true")
    args = parser.parse_args()

    if args.list_skills:
        for name, spec in SKILLS.items():
            print(f"{name:18s} {spec['doc']}")
            print(f"{'':18s} args: {spec['args']}")
        return
    demo()


if __name__ == "__main__":
    main()
