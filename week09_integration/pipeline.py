"""Week 9 - Mini-project pipeline: LLM planner -> skill policy -> safety filter -> robot.

Capstone integration lab. The system under test:

    task text -> [T3 planner] -> skill plan -> [T4 executor] -> robot actions
                                              |-> [T1/T2 safety filter] on every target
    episode stats -> [T5]

Everything runs locally with a deterministic mock planner; a real LLM provider
(same env vars as week 7: LLM_API_KEY / LLM_BASE_URL / LLM_MODEL) is a
going-further option. The low-level skill policy is a P-controller in the spirit
of weeks 6-7; swapping in your week-8 diffusion head is the second going-further
option (adapter stub included as DiffusionPolicyAdapter).

Run:  pixi run --manifest-path ../pixi.toml -e week09 python pipeline.py --episodes 5
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from safety_filter import SafetyFilter, SAFE_BOX

# ----------------------------------------------------------------- skill layer
MAX_DELTA = 0.05   # OSC_POSE maps |action|<=1 to 5 cm per control step
TOL = 0.02
SETTLE = 4
GRIP_STEPS = 30


class RobotAPI:
    """Minimal Lift/Panda API (P-controller), mirroring week 7."""

    def __init__(self, seed: int = 0):
        import robosuite  # deferred: import cost + C:\tmp hints live in week 6/7 labs
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config
        from robosuite.utils import binding_utils
        import mujoco

        # Defensive compatibility patch: compare MuJoCo joint types as plain ints so robosuite's
        # joint-address lookup does not raise an enum-comparison AssertionError (seen on Colab
        # when MuJoCo builds are mixed). Behaviour is identical on a clean MuJoCo 3.4 install.
        def _safe_get_joint_qpos_addr(self, name):
            joint_id = self.joint_name2id(name)
            joint_type = int(self.jnt_type[joint_id])
            joint_addr = self.jnt_qposadr[joint_id]
            ndim = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else (4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
            return joint_addr if ndim == 1 else (joint_addr, joint_addr + ndim)
        binding_utils.MjModel.get_joint_qpos_addr = _safe_get_joint_qpos_addr

        def _safe_get_joint_qvel_addr(self, name):
            joint_id = self.joint_name2id(name)
            joint_type = int(self.jnt_type[joint_id])
            joint_addr = self.jnt_dofadr[joint_id]
            ndim = 6 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else (3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
            return joint_addr if ndim == 1 else (joint_addr, joint_addr + ndim)
        binding_utils.MjModel.get_joint_qvel_addr = _safe_get_joint_qvel_addr

        if os.name == "nt":
            os.makedirs(r"C:\tmp", exist_ok=True)
        cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
        self.env = suite.make(
            env_name="Lift", robots="Panda", controller_configs=cfg,
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
            ignore_done=True, horizon=300, control_freq=20, lite_physics=True,
        )
        self.seed = seed
        self.steps = 0
        self.obs = None

    def reset(self, episode: int):
        np.random.seed(self.seed + episode)
        self.obs = self.env.reset()
        self.steps = 0
        return self.state()

    def state(self) -> dict:
        cube = np.asarray(self.obs["object-state"], dtype=np.float64)[:3]
        return {"eef_pos": np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64),
                "cube_pos": cube,
                "success": bool(self.env._check_success())}

    def _step(self, target, grip):
        self.obs, _, _, _ = self.env.step(self._act(target, grip))
        self.steps += 1
        return np.linalg.norm(target - self.state()["eef_pos"])

    def _act(self, target, grip):
        # OSC_POSE expects actions in [-1, 1] (mapped to +-MAX_DELTA metres),
        # so the position error is normalised, not passed in metres.
        d = np.clip((target - self.state()["eef_pos"]) / MAX_DELTA, -1.0, 1.0)
        a = np.zeros(7); a[:3] = d; a[6] = grip
        return a

    def drive_to(self, target, grip, max_steps=60) -> bool:
        for _ in range(max_steps):
            if self._step(target, grip) < TOL:
                for _ in range(SETTLE):
                    self._step(target, grip)
                return True
            if self.state()["success"]:
                return True
        return False

    def close(self):
        self.env.close()


# ------------------------------------------------------------------ plan layer
PLANS = {
    "pick": [("move_above", {}), ("descend", {}), ("close", {}), ("lift", {})],
    "hover": [("move_above", {})],
    # deliberately unsafe: targets outside the safe box on purpose
    "slam": [("move_to", {"x": 0.0, "y": 0.0, "z": 0.20}), ("wait", {"steps": 5})],
}


def execute_skill(api: RobotAPI, flt: SafetyFilter | None, name: str, args: dict) -> tuple[bool, str, np.ndarray | None]:
    """Run one skill with every geometric target passed through the filter.

    Complete. Returns (ok, reason, target) where target is the ORIGINAL
    unfiltered geometric target (None for 'close'/'wait' which drive no
    geometric goal) - the caller (T4) uses it for near-miss accounting on the
    commanded motion, exactly the quantity a barrier function monitors.
    The 'slam' plan exists to be blocked by the filter: watch the intervention
    counter and the returned reason.
    """
    st = api.state()
    if name == "move_above":
        tgt = st["cube_pos"] + (0, 0, 0.12); grip = -1
    elif name == "descend":
        tgt = st["cube_pos"] + (0, 0, 0.0); grip = -1
    elif name == "close":
        tgt = st["eef_pos"]; grip = 1
        api.drive_to(tgt, grip, GRIP_STEPS)
        return True, "gripper closed", None
    elif name == "lift":
        tgt = st["cube_pos"] + (0, 0, 0.15); grip = 1
    elif name == "move_to":
        tgt = np.array([args["x"], args["y"], args["z"]]); grip = -1
    elif name == "wait":
        return True, f"waited {args.get('steps', 5)}", None
    else:
        return False, f"unknown skill {name!r}", None

    if flt is not None:
        safe = flt.filter_target(tgt)   # near-miss accounting lives in T4 (run_episode)
    else:
        safe = tgt
    reached = api.drive_to(safe, grip)
    return reached, f"{'reached' if reached else 'MISSED'} {np.round(safe, 3).tolist()}", tgt


# --------------------------------------------------------------------- TODOs T3
def plan_task_mock(task: str) -> list[tuple[str, dict]]:
    """TODO (T3): map task text to a plan.

    Rules: if any of 'pick up/grasp/grab/take/lift' in task.lower() -> PLANS['pick'];
    'hover/above' -> PLANS['hover']; 'slam/crush/drop' -> PLANS['slam'];
    otherwise -> PLANS['slam'] (a planner that does nothing sensible is still
    dangerous - the filter is the last line of defence, not the planner).
    Return the plan as a list of (skill, args) tuples.
    """
    raise NotImplementedError("T3: implement plan_task_mock")


# --------------------------------------------------------------------- TODOs T4
def run_episode(api: RobotAPI, flt: SafetyFilter | None, plan, episode: int) -> dict:
    """TODO (T4): execute a plan on one episode, return a result dict.

    Steps: api.reset(episode); iterate plan; execute_skill per step; stop early
    when api.state()['success']; collect: episode, success (bool), steps used,
    interventions (flt.interventions if flt else 0), near_misses
    (flt.near_misses if flt else 0), last skill executed, and a list of the
    (ok, reason) tuples. NOTE: SafetyFilter already counts interventions inside
    filter_target - you only need near_misses: execute_skill returns the
    ORIGINAL unfiltered target as its third value; when it is not None, call
    flt.check_near_miss(target) and increment flt.near_misses if it returns
    True (a near-miss on the *commanded* motion is exactly what a barrier
    monitors - it does not matter that the filter already clamped it).
    """
    raise NotImplementedError("T4: implement run_episode")


# --------------------------------------------------------------------- TODOs T5
def summarise(results: list[dict]) -> None:
    """TODO (T5): print a markdown-style table of the results list:
    | ep | success | steps | interventions | near_misses | last skill |
    followed by a totals line 'X/Y successes, Z filter interventions, W near-misses'.
    Return None. (Solve this even if your episodes are all identical - the
    analysis requires this table format.)"""
    raise NotImplementedError("T5: implement summarise")


class DiffusionPolicyAdapter:
    """Going-further stub: replace the skill layer with your week-8 diffusion head.

    Load ../week08_vla/diffusion_head.pt (proprio) or diffusion_head_vlm.pt
    (VLM-conditioned), rebuild the conditioning, and translate sampled chunks
    into the same 'target, grip' interface used above - or drive the env
    directly with the chunk's first action like week 8's evaluate_policy.
    Keep the SafetyFilter wrapping whatever geometric target you derive.
    """

    def __init__(self, checkpoint: str):
        raise NotImplementedError("going further: adapt week 8's evaluate_policy")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="pick up the cube")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--unsafe", action="store_true", help="bypass the safety filter (comparison run)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default="", help="write results list as JSON here")
    args = p.parse_args()

    api = RobotAPI(seed=args.seed)
    flt = None if args.unsafe else SafetyFilter()
    try:
        plan = plan_task_mock(args.task)
        print(f"task: {args.task!r} -> plan: {[s for s, _ in plan]}"
              f" | filter: {'OFF (unsafe)' if flt is None else 'ON'}")
        results = []
        for ep in range(args.episodes):
            results.append(run_episode(api, flt, plan, ep))
            if flt is not None:
                flt.interventions = 0   # per-episode accounting: reset counters
                flt.near_misses = 0
        summarise(results)
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(results, fh, default=str, indent=2)
    finally:
        api.close()


if __name__ == "__main__":
    main()
