"""Instructor reference solutions for COMP0037 2026/27 Coursework 2 — Question 2.

Patches `validate_plan` and `MobileLLMPlanner` in `assessment/CW02/cw2_starter/q2_mobile_llm_planner.py`,
runs schema & symbolic precondition validation tests, executes nominal and recovery plans
on `MobilePickPlaceEnv` (`PandaOmron`), and saves:
  - `outputs/q2_llm_plan_trace.png`
  - `outputs/q2_llm_rollout.gif`

Usage (from repository root):
    .venv_robot\\Scripts\\python assessment\\CW02\\cw2_solutions\\q2_mobile_llm_solutions.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
STARTER_DIR = os.path.normpath(os.path.join(CUR_DIR, os.pardir, "cw2_starter"))
OUT_DIR = os.path.join(CUR_DIR, "outputs")
sys.path.insert(0, STARTER_DIR)

import mobile_pick_place_env as mpe  # noqa: E402
import q2_mobile_llm_planner as q2  # noqa: E402


def validate_plan(plan: list[dict], initial_state: dict) -> tuple[bool, list[str]]:
    """Reference schema + symbolic precondition validator for MobileSkillLibrary."""
    errors: list[str] = []
    if not isinstance(plan, list) or len(plan) == 0:
        return False, ["Plan must be a non-empty list of skill dictionaries."]

    # Simulate symbolic state transitions
    station = str(initial_state.get("station", "aisle"))
    arm_stowed = bool(initial_state.get("arm_stowed", True))
    holding = bool(initial_state.get("holding_cube", False))

    for i, step in enumerate(plan):
        if not isinstance(step, dict) or "skill" not in step:
            errors.append(f"Step {i}: missing 'skill' key.")
            continue
        sk = step["skill"]
        args = step.get("args", {}) or {}
        if sk not in q2.ALLOWED_SKILLS:
            errors.append(f"Step {i}: unknown skill '{sk}'.")
            continue

        # Validate arguments
        schema = q2.ALLOWED_SKILLS[sk]
        for k, allowed in schema.items():
            if k not in args or args[k] not in allowed:
                errors.append(f"Step {i} ({sk}): invalid arg '{k}'={args.get(k)!r}, expected {allowed}.")

        # Symbolic precondition checks & state updates
        if sk == "stow_arm":
            arm_stowed = True
        elif sk == "drive_to_station":
            if not arm_stowed:
                errors.append(f"Step {i} (drive_to_station): arm must be stowed before driving.")
            target_st = args.get("station")
            if target_st in ("A", "B"):
                station = target_st
        elif sk == "pick_cube":
            if station != "A":
                errors.append(f"Step {i} (pick_cube): robot must be docked at station 'A' (currently '{station}').")
            if holding:
                errors.append(f"Step {i} (pick_cube): gripper is already holding the cube.")
            holding = True
            arm_stowed = True  # pick_cube finishes with lift to safe transit height
        elif sk == "place_cube":
            if station != "B":
                errors.append(f"Step {i} (place_cube): robot must be docked at station 'B' (currently '{station}').")
            if not holding:
                errors.append(f"Step {i} (place_cube): cannot place cube when not holding it.")
            holding = False
            arm_stowed = True
        elif sk == "verify_grasp":
            pass

    return (len(errors) == 0), errors


class MobileLLMPlanner:
    """Reference structured JSON planner with Mock backend (and optional OpenAI/Ollama hook)."""

    def __init__(self, backend: str = "mock") -> None:
        self.backend = backend

    def plan(self, instruction: str, world_state: dict) -> list[dict]:
        text = instruction.lower()
        plan: list[dict] = []
        if not world_state.get("arm_stowed", True):
            plan.append({"skill": "stow_arm", "args": {}})

        if "transfer" in text or ("table a" in text and "table b" in text) or "place" in text:
            if not world_state.get("holding_cube", False):
                if world_state.get("station") != "A":
                    plan.append({"skill": "drive_to_station", "args": {"station": "A"}})
                plan.append({"skill": "pick_cube", "args": {}})
                plan.append({"skill": "verify_grasp", "args": {}})
            if world_state.get("station") != "B":
                plan.append({"skill": "drive_to_station", "args": {"station": "B"}})
            plan.append({"skill": "place_cube", "args": {}})
        elif "pick" in text:
            if world_state.get("station") != "A":
                plan.append({"skill": "drive_to_station", "args": {"station": "A"}})
            plan.append({"skill": "pick_cube", "args": {}})
            plan.append({"skill": "verify_grasp", "args": {}})
        else:
            plan.append({"skill": "drive_to_station", "args": {"station": "A"}})
        return plan


def execute_with_recovery(lib: q2.MobileSkillLibrary, planner: MobileLLMPlanner, instruction: str, max_retries: int = 2) -> dict:
    """Validate and execute a structured plan with closed-loop replanning on skill failure."""
    trace = []
    for attempt in range(max_retries + 1):
        state = lib.get_world_state()
        plan = planner.plan(instruction, state)
        valid, errs = q2.validate_plan(plan, state)
        if not valid:
            return {"success": False, "attempts": attempt + 1, "errors": errs, "trace": trace}

        failed = False
        for step in plan:
            res = lib.execute(step["skill"], step.get("args"))
            trace.append((step["skill"], step.get("args", {}), res.success, res.steps, res.message))
            if not res.success:
                failed = True
                break
        if not failed and lib.get_world_state()["task_success"]:
            return {"success": True, "attempts": attempt + 1, "errors": [], "trace": trace}
    return {"success": lib.get_world_state()["task_success"], "attempts": max_retries + 1, "errors": [], "trace": trace}


def patch_all() -> None:
    q2.validate_plan = validate_plan
    q2.MobileLLMPlanner = MobileLLMPlanner


def main() -> None:
    ap = argparse.ArgumentParser(description="CW2 Q2 LLM Task Planner reference solutions")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    patch_all()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 78)
    print("CW2 Q2 — Typed Mobile-Manipulation Skill API & Precondition Validator")
    print("=" * 78)

    # 1. Precondition validator test battery
    init_state = {"station": "aisle", "arm_stowed": False, "holding_cube": False}
    test_cases = [
        ("Valid full transfer", [
            {"skill": "stow_arm", "args": {}},
            {"skill": "drive_to_station", "args": {"station": "A"}},
            {"skill": "pick_cube", "args": {}},
            {"skill": "verify_grasp", "args": {}},
            {"skill": "drive_to_station", "args": {"station": "B"}},
            {"skill": "place_cube", "args": {}},
        ], True),
        ("Reject: drive before stow", [
            {"skill": "drive_to_station", "args": {"station": "A"}},
        ], False),
        ("Reject: pick from aisle", [
            {"skill": "stow_arm", "args": {}},
            {"skill": "pick_cube", "args": {}},
        ], False),
        ("Reject: place without cube", [
            {"skill": "stow_arm", "args": {}},
            {"skill": "drive_to_station", "args": {"station": "B"}},
            {"skill": "place_cube", "args": {}},
        ], False),
        ("Reject: unknown skill", [
            {"skill": "teleport_cube", "args": {}},
        ], False),
    ]
    passed = 0
    for name, plan, expected in test_cases:
        ok, errs = q2.validate_plan(plan, init_state)
        match = (ok == expected)
        passed += int(match)
        print(f"  [{'PASS' if match else 'FAIL'}] {name:<28} -> valid={ok} (errs={len(errs)})")

    # 2. Closed-loop execution on MobilePickPlaceEnv
    env = mpe.make_mobile_pick_place_env(use_camera_obs=True, camera_size=192, horizon=300, seed=0)
    lib = q2.MobileSkillLibrary(env, record_frames=True)
    lib.reset_env()
    planner = q2.MobileLLMPlanner(backend="mock")
    result = execute_with_recovery(lib, planner, "Transfer the red specimen cube from Table A to Table B.")
    print(f"\nClosed-loop execution: success={result['success']} in {result['attempts']} attempt(s)")
    for sk, sk_args, ok, st, msg in result["trace"]:
        print(f"    - {sk}({sk_args}) -> ok={ok}, steps={st:3d} | {msg}")

    # Plot skill step breakdown
    labels = [f"{sk}\n({a.get('station', '')})" if a else sk for sk, a, _, _, _ in result["trace"]]
    steps_vals = [st for _, _, _, st, _ in result["trace"]]
    cols = ["#16A34A" if ok else "#DC2626" for _, _, ok, _, _ in result["trace"]]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.bar(range(len(labels)), steps_vals, color=cols, edgecolor="#0F172A")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set(ylabel="Control Steps (20 Hz)", title="CW2 Q2: Typed Mobile-Manipulation Skill Execution Trace")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p_out = os.path.join(args.out_dir, "q2_llm_plan_trace.png")
    fig.savefig(p_out, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p_out}")

    # Save rollout GIF
    if lib.frames:
        sampled = lib.frames[::2]
        pil = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE) for f in sampled]
        durs = [90] * len(pil)
        durs[-1] = 1500
        gif_out = os.path.join(args.out_dir, "q2_llm_rollout.gif")
        pil[0].save(gif_out, save_all=True, append_images=pil[1:], duration=durs, loop=0)
        print(f"Saved rollout GIF -> {gif_out} ({len(pil)} frames)")

    env.close()
    print("CW2_Q2_SOLUTIONS_DONE")


if __name__ == "__main__":
    main()

