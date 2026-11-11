"""Week 05 - Part A: MuJoCo CPU baseline benchmark.

Measures how many physics steps per second plain (CPU) MuJoCo can run on
your machine, one environment at a time. This is the number the batched
GPU benchmark (benchmark_mujoco_warp.py) has to beat.

TODO T1 is the only thing you implement here: a wall-clock timing loop
around mj_step.

Run:
    pixi run -e week05 python week05_simulation/benchmark_mujoco_cpu.py
    pixi run -e week05 python week05_simulation/benchmark_mujoco_cpu.py --steps 20000
"""

import argparse
import time

from sim_models import cartpole_model


def rollout_cpu(m, n_steps, ctrl=0.0):
    """Advance a single MuJoCo environment for n_steps physics steps.

    Uses the plain mujoco package on CPU. Returns the final state
    (qpos, qvel) so callers can sanity-check that physics actually ran.
    """
    import mujoco

    d = mujoco.MjData(m)
    d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d)
    for _ in range(n_steps):
        mujoco.mj_step(m, d)
    return d.qpos.copy(), d.qvel.copy()


def time_cpu_rollout(m, n_steps, n_repeats=5):
    """Time the CPU rollout and return steps-per-second (best of n_repeats).

    TODO (T1): implement the timing loop.
      1. Call rollout_cpu(m, n_steps) once and IGNORE the result - the first
         call warms caches and may be slower.
      2. For each of the n_repeats repeats, time one rollout_cpu call with
         time.perf_counter() around it.
      3. steps/sec = n_steps / fastest elapsed seconds.
    Return that best-case steps/sec as a float.
    """
    raise NotImplementedError("T1: implement the CPU timing loop")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo CPU baseline benchmark")
    parser.add_argument("--steps", type=int, default=5000, help="physics steps per repeat")
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats (best kept)")
    args = parser.parse_args()

    m = cartpole_model()
    qpos, qvel = rollout_cpu(m, 10)
    print(f"warm-up rollout ok: qpos={qpos.round(3)} qvel={qvel.round(3)}")

    try:
        sps = time_cpu_rollout(m, args.steps, args.repeats)
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    complete T1 in benchmark_mujoco_cpu.py, then re-run.")
        return 1

    print(f"\nMuJoCo CPU baseline (single environment, {args.steps} steps/rollout)")
    print(f"  best of {args.repeats}: {sps:,.0f} steps/s")
    print("  -> note this number: it is the baseline for the Warp benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
