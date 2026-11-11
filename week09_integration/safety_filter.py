"""Week 9 - Safety filter for end-effector commands (CBF-style barrier, deployable).

The high-level planner emits absolute end-effector targets. Before anything is
sent to the robot, each target passes through this filter. You implement a
control-barrier-function-style projection: a hard workspace box with a safety
margin, plus an h() that reports how close the *commanded* motion is to each
boundary (used by the pipeline to log near-misses and to interlock).

This is the same idea as week 4's joint-limit shield, one level up: week 4
protected joint angles, this protects the end-effector workspace (keep the
gripper above the table, inside the reachable box, away from the arena walls).
"""

from __future__ import annotations

import numpy as np

# Safe workspace for the Panda in robosuite Lift (metres, world frame).
# x/y match week 7's WORKSPACE_LIMITS; the z floor sits just above the table
# surface (~0.81) so the gripper can still descend to cube centre (~0.83) but
# can never be commanded into the table. h_min is subtracted on every side
# in the filter below, so the effective z floor is 0.79 + h_min = 0.82.
SAFE_BOX = {
    "x": (-0.25, 0.25),
    "y": (-0.35, 0.15),
    "z": (0.79, 1.12),
}


class SafetyFilter:
    """Projects commanded eef targets into the safe box and reports margins."""

    def __init__(self, h_min: float = 0.03, box: dict | None = None):
        self.h_min = h_min
        self.box = box if box is not None else SAFE_BOX
        self.interventions = 0   # commands that had to be modified
        self.near_misses = 0      # commands left inside but within h_min of a boundary

    # ------------------------------------------------------------------ TODO T1
    def filter_target(self, target: np.ndarray) -> np.ndarray:
        """Return a safe version of `target` (3,) world-frame position.

        TODO (T1): for each axis, clamp the coordinate into
        [box[axis][0] + h_min, box[axis][1] - h_min].
        If the returned target differs from the input by more than 1e-9,
        increment self.interventions.  Return a NEW array - never edit the input.
        """
        raise NotImplementedError("T1: implement filter_target")

    # ------------------------------------------------------------------ TODO T2
    def h(self, target: np.ndarray) -> dict[str, float]:
        """Barrier values: signed distance from `target` to each box boundary.

        TODO (T2): return {"x-": dist to lower x wall, "x+": dist to upper x wall,
        "y-", "y+", "z-", "z+": ...} where dist = (target[axis] - lower) for minus
        walls and (upper - target[axis]) for plus walls.  Do NOT clamp here - the
        caller uses these values to detect near-misses on the ORIGINAL command.
        """
        raise NotImplementedError("T2: implement h")

    def check_near_miss(self, original: np.ndarray) -> bool:
        """True if any barrier of the *original* command is inside h_min.

        Complete - uses your h(). The pipeline calls this before filtering so
        that 'we nearly hit a wall' is visible even when no clamp was needed.
        """
        return any(v < self.h_min for v in self.h(original).values())


def self_test() -> None:
    """Sanity checks for the reference behaviour (run after implementing)."""
    f = SafetyFilter()
    t = np.array([0.5, -0.5, 0.5])
    s = f.filter_target(t)
    assert s[0] <= SAFE_BOX["x"][1] - f.h_min + 1e-9, "x not clamped"
    assert s[2] >= SAFE_BOX["z"][0] + f.h_min - 1e-9, "z floor not enforced"
    assert f.interventions == 1, "intervention not counted"
    inner = f.filter_target(np.array([0.0, 0.0, 0.95]))
    assert f.interventions == 1, "clean target should not count as intervention"
    assert not f.check_near_miss(inner), "mid-box target is not a near-miss"
    print("safety_filter self-test OK - safe:", s.round(3).tolist())


if __name__ == "__main__":
    self_test()
