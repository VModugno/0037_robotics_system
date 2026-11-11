"""Week 07 - LLMs in robotics, part 2: the LLM task planner.

Pipeline:  natural-language task
              -> prompt (skill schema + world state)     [T1]
              -> LLM (any OpenAI-compatible / Ollama / offline mock)
              -> raw text
              -> JSON plan                              [T2]
              -> validation against the skill registry  [T3]
              -> execution on the robot                 [T4]

The planner is *provider-agnostic* on purpose. An LLM, from the robot's point
of view, is a function from text to text; everything else -- prompting,
parsing, validation, execution -- is robotics engineering. Swap
`--provider mock|openai|ollama` and nothing else in the pipeline changes.

TODO map (this file):
    T1  build_messages        the prompt: skill schema + state + task
    T2  extract_json          robust JSON extraction from raw LLM text
    T3  validate_plan         schema + workspace safety checks
    T4  execute_plan          the executor: run steps, handle errors

Usage:
    # offline, no API key, no LLM (keyword mock -- good for the pipeline):
    pixi run --manifest-path ../pixi.toml -e week07 python llm_planner.py \
        --provider mock --task "pick up the cube"

    # dry-run: plan + validation only, no robot (fast, works anywhere):
    ... --dry-run

    # real LLMs (env vars LLM_API_KEY / LLM_MODEL / LLM_BASE_URL):
    ... --provider openai --task "grasp the cube and lift it high"
    ... --provider ollama --task "lift the cube"   # local Ollama server
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from robot_api import SKILLS, RobotSkillAPI, call_skill


# --------------------------------------------------------------------------- #
# TODO T1: the prompt
# --------------------------------------------------------------------------- #


def build_messages(task, state):
    """TODO T1. Build the chat messages sent to the LLM.

    Return a list of {"role", "content"} dicts, OpenAI-style:
      * a system message that (a) describes the robot, (b) lists EVERY skill
        from the SKILLS registry with its doc and args (build this from the
        registry -- do not hardcode skill descriptions), (c) specifies the
        exact output format: a JSON object
            {"plan": [{"skill": <name>, "args": {...}}, ...]}
        and nothing else -- no prose, no markdown fences;
      * a user message containing the current world state (from `state`) and
        the task.

    A model that receives a precise tool list and a machine-checkable output
    contract fails far less often than one asked to "just write the plan".
    """
    raise NotImplementedError("T1: build the system + user messages (see docstring)")


# --------------------------------------------------------------------------- #
# Providers (complete) -- an LLM is just a text->text function over HTTP
# --------------------------------------------------------------------------- #


class MockLLM:
    """A keyword 'LLM': zero cost, deterministic, offline.

    Not intelligent -- and that is the point. It stands in the exact place a
    real model occupies, so the whole pipeline can be graded offline. Its
    plans are written to be *plausible but imperfect* (see the fallback).
    """

    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        task = messages[-1]["content"].lower()
        plan = []
        if any(w in task for w in ("pick up", "pick-up", "grasp", "grab", "take", "lift")):
            plan = [
                {"skill": "move_above", "args": {"target": "cube", "dz": 0.12}},
                {"skill": "descend_to_cube", "args": {"dz": 0.0}},
                {"skill": "close_gripper", "args": {}},
                {"skill": "lift", "args": {"height": 0.15}},
            ]
            if "high" in task:
                plan.append({"skill": "lift", "args": {"height": 0.10}})
        elif "hover" in task or "above" in task:
            plan = [{"skill": "move_above", "args": {"target": "cube", "dz": 0.12}}]
        elif "release" in task or "open" in task:
            plan = [{"skill": "open_gripper", "args": {}}]
        elif "close" in task:
            plan = [
                {"skill": "descend_to_cube", "args": {"dz": 0.0}},
                {"skill": "close_gripper", "args": {}},
            ]
        else:
            # the mock "misunderstands": a plausible-looking but useless plan.
            # The executor will report it; the fix is a better model OR a
            # better prompt OR a validation pass -- all three are real levers.
            plan = [
                {"skill": "move_to", "args": {"x": 0.0, "y": 0.0, "z": 1.05}},
                {"skill": "wait", "args": {"steps": 2}},
            ]
        return json.dumps({"plan": plan})


def _post_json(url, payload, headers, timeout=60):
    """POST a JSON payload, return the parsed response (stdlib only)."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def openai_chat(messages, model, base_url, api_key):
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, vLLM, LM Studio...)."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": 0.0}
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = _post_json(url, payload, headers)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"LLM endpoint HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
        ) from None
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"could not reach {url} ({e.reason}). Is the server running / "
            "LLM_BASE_URL set correctly?"
        ) from None
    return resp["choices"][0]["message"]["content"]


def ollama_chat(messages, model, base_url, api_key=None):
    """A local Ollama server (POST /api/chat)."""
    url = base_url.rstrip("/") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    resp = _post_json(url, payload, {})
    return resp["message"]["content"]


# --------------------------------------------------------------------------- #
# TODO T2: parsing
# --------------------------------------------------------------------------- #


def extract_json(text):
    """TODO T2. Pull the JSON plan object out of raw LLM text.

    LLMs wrap JSON in prose and ```json fences no matter what you ask.
    Strategy: find the FIRST '{' and the LAST '}' in the text and parse the
    substring with json.loads. Raise a ValueError (with a helpful message) if
    parsing fails, or return None if the text contains no braces at all.

    Returns the parsed object (dict).
    """
    raise NotImplementedError("T2: extract the JSON object (see docstring)")


# --------------------------------------------------------------------------- #
# TODO T3: validation -- the first safety layer
# --------------------------------------------------------------------------- #


def validate_plan(plan, api):
    """TODO T3. Check the plan against the skill registry and the workspace.

    Returns (ok: bool, errors: list[str]).

    For every step of plan["plan"]:
      * the step is a dict with a "skill" key and an "args" dict;
      * the skill name exists in the SKILLS registry;
      * every key in args is a declared arg of that skill, and every
        declared arg is provided;
      * numeric args of move_to lie inside the workspace box from
        api.WORKSPACE_LIMITS (a plan that would drive the arm outside its
        envelope must be refused BEFORE execution).

    Collect ALL errors (not just the first) and report them with the step
    index, e.g. "step 2: unknown skill 'teleport'".
    """
    raise NotImplementedError("T3: validate the plan (see docstring)")


# --------------------------------------------------------------------------- #
# TODO T4: execution
# --------------------------------------------------------------------------- #


def execute_plan(api, plan, verbose=True):
    """TODO T4. Execute a validated plan step by step.

    For each step, call the skill via call_skill(api, name, args) and print
    the result. If a skill returns "error", print the state, STOP, and return
    (False, step_index). If all steps run, return (True, n_steps).
    Optionally break early once api.state()["success"] is True and say so.

    The executor is the robot's last line of defence: a validated plan can
    still fail in the physical world (the cube moved). Handling that
    gracefully -- stop, report, do NOT improvise -- is the safety culture
    this week is teaching.
    """
    raise NotImplementedError("T4: execute the plan (see docstring)")


# --------------------------------------------------------------------------- #
# Driver (complete)
# --------------------------------------------------------------------------- #


class _DryRunAPI:
    """Workspace bounds only -- validate_plan needs nothing else from the API."""

    WORKSPACE_LIMITS = RobotSkillAPI.WORKSPACE_LIMITS


def plan_task(provider, task, seed, dry_run, model, base_url, api_key):
    if provider == "mock":
        llm = MockLLM()
        chat = lambda msgs: llm.chat(msgs)
    elif provider == "openai":
        chat = lambda msgs: openai_chat(msgs, model, base_url, api_key)
    elif provider == "ollama":
        chat = lambda msgs: ollama_chat(msgs, model, base_url, api_key)
    else:
        raise SystemExit(f"unknown provider {provider!r}")

    print(f"-- provider: {provider}" + (f" ({model})" if provider != "mock" else ""))
    print(f"-- task: {task!r}")

    if dry_run:
        state = {
            "eef_pos": [0.0, 0.0, 1.0],
            "cube_pos": [0.0, 0.0, 0.83],
            "gripper_open": True,
            "success": False,
        }
        api = None
    else:
        api = RobotSkillAPI(seed=seed)
        state = api.state()
    print(f"-- world state: {state}")

    try:
        messages = build_messages(task, state)
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    Complete TODO T1 in llm_planner.py, then re-run.")
        return 1

    try:
        raw = chat(messages)
    except RuntimeError as e:
        print(f"\n[!] {e}")
        return 1
    print(f"-- raw LLM output:\n{raw}")

    try:
        plan = extract_json(raw)
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    Complete TODO T2 in llm_planner.py, then re-run.")
        return 1
    if plan is None:
        print("[!] the model produced no JSON object. Improve T1's prompt "
              "(be more explicit about the output format) and retry.")
        return 1

    print("-- parsed plan:")
    for i, step in enumerate(plan.get("plan", [])):
        print(f"   {i}: {step}")

    target = api if api is not None else _DryRunAPI()
    try:
        ok, errors = validate_plan(plan, target)
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    Complete TODO T3 in llm_planner.py, then re-run.")
        return 1
    if not ok:
        print("[!] plan REJECTED by validation:")
        for err in errors:
            print(f"    - {err}")
        return 1
    print("-- plan validated")

    if dry_run:
        print("-- dry run: skipping execution")
        return 0

    try:
        done, last = execute_plan(api, plan)
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    Complete TODO T4 in llm_planner.py, then re-run.")
        return 1
    final = api.state()
    print(f"-- final state: {final}")
    api.close()
    if final["success"]:
        print("TASK SUCCESS")
        return 0
    print(f"TASK FAILED (stopped at step {last if not done else 'end'})")
    return 1


def main():
    parser = argparse.ArgumentParser(description="Week 07 LLM task planner")
    parser.add_argument("--provider", choices=["mock", "openai", "ollama"], default="mock")
    parser.add_argument("--task", type=str, default="pick up the cube")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="plan + validate only")
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key", type=str, default=os.environ.get("LLM_API_KEY", ""))
    args = parser.parse_args()

    rc = plan_task(
        provider=args.provider,
        task=args.task,
        seed=args.seed,
        dry_run=args.dry_run,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
