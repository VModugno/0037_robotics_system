"""COMP0037 - Week 07 instructor solutions (LLM task planning).

Reference implementations for the four student TODOs in
week07_llm_planning/llm_planner.py:

    T1  build_messages   system prompt built FROM the SKILLS registry
    T2  extract_json     first '{' .. last '}' JSON extraction
    T3  validate_plan    registry + workspace-bounds checks, all errors
    T4  execute_plan     step-by-step execution, stop-on-error, early success

Run (from labs/) after each stage or all at once:

    pixi run --manifest-path pixi.toml -e week07 python solutions/week07_llm_planning_solutions.py --stage all

Stages:
    checks   unit checks for T1/T2/T3 on a dry-run API (no env, seconds)
    dryrun   llm_planner.main --provider mock --dry-run  (no env)
    e2e      llm_planner.main --provider mock on the real Lift env
             (expects TASK SUCCESS, ~1-2 min)
    all      checks -> dryrun -> e2e
"""

import argparse
import json
import os
import sys

LAB_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "week07_llm_planning")
)
sys.path.insert(0, LAB_DIR)

import llm_planner  # noqa: E402  (needs LAB_DIR on sys.path)
from robot_api import SKILLS  # noqa: E402


# --------------------------------------------------------------------------- #
# Reference T1: the prompt, built from the registry
# --------------------------------------------------------------------------- #


def build_messages(task, state):
    """System: robot description + every skill (from SKILLS) + output contract.

    User: world state + task. Built from the registry so the prompt and the
    API can never drift apart.
    """
    lines = []
    for name, spec in SKILLS.items():
        args = ", ".join(f"{a}: {t}" for a, t in spec["args"].items()) or "none"
        lines.append(f"- {name}({args}): {spec['doc']}")
    skills_text = "\n".join(lines)

    system = (
        "You are the planning module of a robotic manipulator (a Panda arm "
        "with a parallel-jaw gripper operating on a table). You do not "
        "control the robot directly: you emit plans made of skill calls.\n\n"
        "Available skills:\n"
        f"{skills_text}\n\n"
        "Answer with a single JSON object of the exact form\n"
        '{"plan": [{"skill": "<name>", "args": {<args>}}, ...]}\n'
        "listing the calls in execution order. Use only declared args, "
        "provide every declared arg, and output JSON only -- no prose, no "
        "markdown fences, no comments."
    )
    user = f"World state: {json.dumps(state)}\nTask: {task}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Reference T2: robust JSON extraction
# --------------------------------------------------------------------------- #


def extract_json(text):
    """First '{' to last '}'; ValueError on bad JSON, None if no braces."""
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    blob = text[first : last + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse LLM output as JSON: {e}\n{blob[:200]}") from None


# --------------------------------------------------------------------------- #
# Reference T3: validation -- registry + workspace safety
# --------------------------------------------------------------------------- #


def validate_plan(plan, api):
    """Collect ALL errors across every step (never early-return on one)."""
    errors = []
    steps = plan.get("plan", []) if isinstance(plan, dict) else []
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "skill" not in step or "args" not in step:
            errors.append(f"step {i}: not an object with 'skill' and 'args'")
            continue
        name = step["skill"]
        args = step["args"]
        if name not in SKILLS:
            errors.append(f"step {i}: unknown skill {name!r}")
            continue
        declared = SKILLS[name]["args"]
        if not isinstance(args, dict):
            errors.append(f"step {i}: 'args' must be an object")
            continue
        missing = sorted(set(declared) - set(args))
        extra = sorted(set(args) - set(declared))
        if missing:
            errors.append(f"step {i}: {name} missing args {missing}")
        if extra:
            errors.append(f"step {i}: {name} got unexpected args {extra}")
        # workspace-bounds check for the only skill with raw world coords
        if name == "move_to" and not missing and not extra:
            for axis, (lo, hi) in api.WORKSPACE_LIMITS.items():
                v = float(args[axis])
                if not (lo - 1e-9 <= v <= hi + 1e-9):
                    errors.append(
                        f"step {i}: move_to {axis}={v:.3f} outside workspace "
                        f"[{lo:.2f}, {hi:.2f}]"
                    )
    return len(errors) == 0, errors


# --------------------------------------------------------------------------- #
# Reference T4: execution -- stop on error, early-exit on success
# --------------------------------------------------------------------------- #


def execute_plan(api, plan, verbose=True):
    """Run the plan; (True, n) if complete, (False, i) where it stopped."""
    steps = plan.get("plan", [])
    for i, step in enumerate(steps):
        name, args = step["skill"], step["args"]
        status, info = llm_planner.call_skill(api, name, args)
        if verbose:
            print(f"[plan {i}] {name} {args} -> {status} {info}")
        if status == "error":
            print(f"[!] skill failed at step {i}: {info}")
            print(f"    state: {api.state()}")
            return False, i
        if api.state()["success"]:
            if verbose:
                print(f"[i] task succeeded early -- stopping after step {i}")
            return True, i + 1
    return True, len(steps)


# --------------------------------------------------------------------------- #
# Patch + verification stages
# --------------------------------------------------------------------------- #


def patch_all():
    llm_planner.build_messages = build_messages
    llm_planner.extract_json = extract_json
    llm_planner.validate_plan = validate_plan
    llm_planner.execute_plan = execute_plan


class _DryAPI:
    WORKSPACE_LIMITS = llm_planner.RobotSkillAPI.WORKSPACE_LIMITS


def run_checks():
    print("=" * 70)
    print("stage: unit checks (T1/T2/T3) -- no env")
    print("=" * 70)
    ok = True

    # T1: registry-driven prompt
    msgs = build_messages("pick up the cube", {"eef_pos": [0, 0, 1], "cube_pos": [0, 0, 0.83]})
    sys_msg = msgs[0]["content"]
    missing = [n for n in SKILLS if n not in sys_msg]
    assert len(msgs) == 2 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert not missing, f"T1 omits skills: {missing}"
    assert '"plan"' in sys_msg, "T1 lacks the output contract"
    print(f"  T1 ok: system prompt lists all {len(SKILLS)} skills + contract")

    # T2: fenced / prose-wrapped JSON, bad JSON, no JSON
    obj = extract_json('Sure! ```json\n{"plan": [{"skill": "lift", "args": {"height": 0.15}}]}\n```')
    assert obj["plan"][0]["skill"] == "lift", "T2 fence case"
    assert extract_json("no braces here") is None, "T2 no-brace case"
    try:
        extract_json('{"plan": [broken}')
        raise AssertionError("T2 should raise ValueError on bad JSON")
    except ValueError:
        pass
    print("  T2 ok: fences, no-brace -> None, bad JSON -> ValueError")

    # T3: a plan with every error class at once
    bad = {
        "plan": [
            {"skill": "teleport", "args": {"x": 0}},                       # unknown
            {"skill": "move_to", "args": {"x": 0.0, "y": 0.0}},            # missing z
            {"skill": "move_to", "args": {"x": 9.9, "y": 0.0, "z": 1.0}},  # out of bounds
            {"skill": "lift", "args": {"height": 0.1, "warp": 1}},         # extra arg
            "not-a-step",                                                    # not a dict
        ]
    }
    okflag, errors = validate_plan(bad, _DryAPI)
    assert not okflag and len(errors) == 5, f"expected 5 errors, got {errors}"
    good = {"plan": [{"skill": "move_to", "args": {"x": 0.1, "y": 0.1, "z": 0.9}}]}
    okflag2, errors2 = validate_plan(good, _DryAPI)
    assert okflag2 and not errors2, f"valid plan rejected: {errors2}"
    print(f"  T3 ok: bad plan -> {len(errors)} collected errors; good plan accepted")

    print("CHECKS PASS")
    return True


def run_main(argv):
    """Call llm_planner.main with a fake argv; return its exit code."""
    old = sys.argv
    try:
        sys.argv = ["llm_planner.py"] + argv
        llm_planner.main()
    except SystemExit as e:
        return int(e.code or 0)
    finally:
        sys.argv = old
    return 0


def run_dryrun():
    print("=" * 70)
    print("stage: mock provider dry-run (plan + validate, no robot)")
    print("=" * 70)
    rc = run_main(["--provider", "mock", "--task", "pick up the cube", "--dry-run"])
    assert rc == 0, f"dry-run exited {rc}"
    print("DRYRUN PASS (rc=0)")


def run_e2e():
    print("=" * 70)
    print("stage: mock provider end-to-end on the real Lift env")
    print("=" * 70)
    rc = run_main(["--provider", "mock", "--task", "pick up the cube", "--seed", "0"])
    assert rc == 0, f"end-to-end exited {rc} (expected TASK SUCCESS, rc=0)"
    print("E2E PASS (rc=0, TASK SUCCESS)")


def main():
    parser = argparse.ArgumentParser(description="Week 07 solutions runner")
    parser.add_argument(
        "--stage", choices=["checks", "dryrun", "e2e", "all"], default="all"
    )
    args = parser.parse_args()

    patch_all()

    if args.stage in ("checks", "all"):
        run_checks()
    if args.stage in ("dryrun", "all"):
        run_dryrun()
    if args.stage in ("e2e", "all"):
        run_e2e()
    print("\nWEEK07 SOLUTIONS: ALL STAGES PASS")


if __name__ == "__main__":
    main()
