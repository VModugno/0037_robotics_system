"""Week 09 - Instructor solutions: mini-project pipeline (planner -> skills -> CBF filter -> robot).

Run from the week09_integration folder (outputs land in cwd):

    pixi run --manifest-path ../pixi.toml -e week09 python ../solutions/week09_integration_solutions.py

Reference implementations for TODOs T1-T5, then the real pipeline main() is
executed with the patches applied. Stages:

    selftest   safety_filter unit checks (instant)
    pick       pipeline main, task 'pick up the cube', filter ON (5 episodes)
    slam       pipeline main, task 'slam the cube', filter ON (blocked by the filter)
    unsafe     pipeline main, task 'slam the cube', --unsafe (no filter: raw command)
    all        selftest + pick + slam + unsafe (reference run for LAB.md numbers)
"""

from __future__ import annotations

import os
import sys

LAB_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "week09_integration"))
sys.path.insert(0, LAB_DIR)

import numpy as np  # noqa: E402

import pipeline as pl  # noqa: E402
import safety_filter as sf  # noqa: E402


# ------------------------------------------------------------------ T1 / T2
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _filter_target(self, target: np.ndarray) -> np.ndarray:
    """Clamp each axis into [lower + h_min, upper - h_min]; count interventions."""
    t = np.asarray(target, dtype=np.float64).copy()
    modified = False
    for axis, (lo, hi) in self.box.items():
        i = AXIS_INDEX[axis]
        lo_s, hi_s = lo + self.h_min, hi - self.h_min
        if t[i] < lo_s:
            t[i] = lo_s
            modified = True
        elif t[i] > hi_s:
            t[i] = hi_s
            modified = True
    if modified:
        self.interventions += 1
    return t


def _h(self, target: np.ndarray) -> dict:
    """Signed distance from the ORIGINAL command to each box boundary."""
    t = np.asarray(target, dtype=np.float64)
    out = {}
    for axis, (lo, hi) in self.box.items():
        i = AXIS_INDEX[axis]
        out[axis + "-"] = float(t[i] - lo)
        out[axis + "+"] = float(hi - t[i])
    return out


# ------------------------------------------------------------------ T3
def _plan_task_mock(task: str):
    t = task.lower()
    if any(k in t for k in ("pick up", "grasp", "grab", "take", "lift")):
        return pl.PLANS["pick"]
    if any(k in t for k in ("hover", "above")):
        return pl.PLANS["hover"]
    if any(k in t for k in ("slam", "crush", "drop")):
        return pl.PLANS["slam"]
    # unknown task -> the dangerous fallback: the filter is the last line of defence
    return pl.PLANS["slam"]


# ------------------------------------------------------------------ T4
def _run_episode(api, flt, plan, episode: int) -> dict:
    api.reset(episode)
    reasons = []
    last_skill = ""
    for name, args in plan:
        last_skill = name
        ok, reason, target = pl.execute_skill(api, flt, name, args)
        reasons.append((ok, reason))
        if flt is not None and target is not None and flt.check_near_miss(target):
            flt.near_misses += 1
        if not ok:
            break
        if api.state()["success"]:
            break
    st = api.state()
    return {
        "episode": episode,
        "success": bool(st["success"]),
        "steps": api.steps,
        "interventions": flt.interventions if flt is not None else 0,
        "near_misses": flt.near_misses if flt is not None else 0,
        "last skill": last_skill,
        "reasons": reasons,
    }


# ------------------------------------------------------------------ T5
def _summarise(results) -> None:
    print("| ep | success | steps | interventions | near_misses | last skill |")
    print("|----|---------|-------|---------------|-------------|------------|")
    for r in results:
        print(f"| {r['episode']} | {r['success']} | {r['steps']} | "
              f"{r['interventions']} | {r['near_misses']} | {r['last skill']} |")
    succ = sum(1 for r in results if r["success"])
    inter = sum(r["interventions"] for r in results)
    near = sum(r["near_misses"] for r in results)
    print(f"{succ}/{len(results)} successes, {inter} filter interventions, {near} near-misses")


# ------------------------------------------------------------------ runner
def patch_all() -> None:
    sf.SafetyFilter.filter_target = _filter_target
    sf.SafetyFilter.h = _h
    pl.plan_task_mock = _plan_task_mock
    pl.run_episode = _run_episode
    pl.summarise = _summarise


def run_selftest() -> None:
    sf.self_test()


def run_pipeline(argv: list[str]) -> None:
    sys.argv = ["pipeline.py"] + argv
    pl.main()


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    extra = sys.argv[2:]
    patch_all()

    if stage in ("selftest", "all"):
        print("-- stage: selftest " + "-" * 40)
        run_selftest()
    if stage in ("pick", "all"):
        print("-- stage: pick (filter ON) " + "-" * 40)
        run_pipeline(["--task", "pick up the cube", "--episodes", "5"] + extra)
    if stage in ("slam", "all"):
        print("-- stage: slam (filter ON) " + "-" * 40)
        run_pipeline(["--task", "slam the cube", "--episodes", "2"] + extra)
    if stage in ("unsafe", "all"):
        print("-- stage: slam (filter OFF) " + "-" * 40)
        run_pipeline(["--task", "slam the cube", "--episodes", "2", "--unsafe"] + extra)


if __name__ == "__main__":
    main()
