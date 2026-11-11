"""Week 05 - Part B: MuJoCo Warp batched GPU benchmark.

MuJoCo Warp (mujoco-warp) runs `nworld` independent physics worlds in ONE
step call on the GPU. This script sweeps the world count and measures
total throughput (steps x worlds / second), then compares it to the CPU
baseline from Part A.

You implement three things:
  T2 - build the batched model + data for a given world count
  T3 - time the batched step loop (CUDA-graph capture on GPU, plain loop on CPU)
  T4 - check the overflow flags after a run

Runs on CPU too (warp device "cpu") - expect it to be SLOWER than Part A:
on CPU you pay batching overhead with zero parallelism. That contrast is
itself a lab observation, not a bug.

Run:
    pixi run -e week05 python week05_simulation/benchmark_mujoco_warp.py
    pixi run -e week05 python week05_simulation/benchmark_mujoco_warp.py \
        --nworlds 1,2,4,8,16 --steps 2000 --cpu-baseline 250000
"""

import argparse

import numpy as np

try:
    import warp as wp
except ImportError as e:  # pragma: no cover
    raise SystemExit("mujoco-warp needs warp-lang: check `pixi run -e week05` install") from e

import mujoco_warp as mjw

from sim_models import cartpole_model


def device_is_cuda() -> bool:
    return wp.get_device().is_cuda


def describe_device():
    """Return the warp device string, or exit with a readable cause on failure."""
    try:
        return str(wp.get_device())
    except (RuntimeError, OSError) as e:
        raise SystemExit(
            f"[!] warp runtime failed to initialise:\n    {e}\n"
            "    On managed Windows machines an Application Control policy often\n"
            "    blocks warp\\bin\\warp-clang.dll (WinError 4551). Run this lab on\n"
            "    Colab (see LAB.md) or on a machine where that DLL can load."
        ) from e


def make_batched(mjm, nworld, seed=0):
    """Build a batched model and data for `nworld` worlds.

    TODO (T2): implement using the MuJoCo Warp API.
      1. model = mjw.put_model(mjm)                    # upload to device
      2. data  = mjw.make_data(mjm, nworld=nworld)     # batched data buffer
      3. give each world a slightly different initial pole velocity so the
         worlds are not identical:
             qvel = data.qvel.numpy()                  # (nworld, nv)
             rng = np.random.default_rng(seed)
             qvel[:, 1] = rng.uniform(-1.0, 1.0, size=nworld)
             data.qvel = wp.array(qvel, dtype=wp.float32)
      4. mjw.forward(model, data)                      # recompute derived state
    Return (model, data).
    """
    raise NotImplementedError("T2: implement make_batched")


def time_batched_steps(model, data, n_steps):
    """Time n_steps batched steps; return total steps/sec (steps x worlds / s).

    TODO (T3): implement the timing protocol.
      1. Warm-up: one mjw.step(model, data) call - the FIRST call JIT-compiles
         kernels and can take seconds. Never time it.
      2. If wp.get_device().is_cuda: capture a CUDA graph
             with wp.ScopedCapture() as capture:
                 mjw.step(model, data)
         then call wp.capture_launch(capture.graph) once to finish compiling,
         wp.synchronize(), and time a loop of wp.capture_launch(capture.graph)
         calls. Without graph capture, one step fires hundreds of tiny kernel
         launches and Python dispatch dominates the measurement.
      3. On CPU: just time a plain loop of mjw.step(model, data).
      4. Call wp.synchronize() BEFORE stopping the clock (steps launch async).
      5. throughput = n_steps * nworld / elapsed_seconds, where
         nworld = data.qpos.shape[0].
    Return that float.
    """
    raise NotImplementedError("T3: implement time_batched_steps")


def check_overflow(data):
    """Verify no world hit the fixed-size contact/constraint buffers.

    TODO (T4): implement.
      data.overflow is a per-world bitmask (mjw.OverflowType). Convert to
      numpy with .numpy(); if ANY bit is nonzero, return the world indices
      that overflowed as an int array. Return an empty array when clean.
    """
    raise NotImplementedError("T4: implement check_overflow")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo Warp batched benchmark")
    parser.add_argument("--nworlds", type=str, default="1,2,4,8,16,32,64",
                        help="comma-separated world counts to sweep")
    parser.add_argument("--steps", type=int, default=2000, help="physics steps per timing run")
    parser.add_argument("--cpu-baseline", type=float, default=None,
                        help="steps/sec from Part A to draw as reference line")
    parser.add_argument("--show", action="store_true", help="open the plot window")
    args = parser.parse_args()

    nworlds = [int(n) for n in args.nworlds.split(",")]
    print(f"warp device: {describe_device()}")
    if not device_is_cuda():
        print("[i] CPU device: numbers are for the batching-overhead observation,")
        print("    not for the GPU speedup. Re-run on Colab T4 for the real thing.")

    mjm = cartpole_model()
    results = {}

    for nworld in nworlds:
        try:
            model, data = make_batched(mjm, nworld)
        except NotImplementedError as e:
            print(f"\n[!] {e}")
            print("    complete T2 in benchmark_mujoco_warp.py, then re-run.")
            return 1
        try:
            sps = time_batched_steps(model, data, args.steps)
        except NotImplementedError as e:
            print(f"\n[!] {e}")
            print("    complete T3 in benchmark_mujoco_warp.py, then re-run.")
            return 1
        try:
            bad = check_overflow(data)
        except NotImplementedError as e:
            print(f"\n[!] {e}")
            print("    complete T4 in benchmark_mujoco_warp.py, then re-run.")
            return 1
        results[nworld] = sps
        flag = f"  OVERFLOW in worlds {bad.tolist()}" if len(bad) else ""
        print(f"  nworld={nworld:>4}: {sps:>14,.0f} total steps/s{flag}")

    print("\nsummary (worlds -> total steps/s, best-effort throughput):")
    for nworld, sps in results.items():
        per_world = sps / nworld
        print(f"  {nworld:>4} worlds: {sps:>14,.0f} total  ({per_world:>12,.0f}/world)")

    plot_results(results, args.cpu_baseline, args.show)
    return 0


def plot_results(results, cpu_baseline, show):
    import matplotlib.pyplot as plt

    ns = sorted(results)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, [results[n] for n in ns], "o-", label="MuJoCo Warp (batched)")
    if cpu_baseline:
        ax.axhline(cpu_baseline, color="tab:red", ls="--",
                   label=f"MuJoCo CPU single-env ({cpu_baseline:,.0f}/s)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of parallel worlds (nworld)")
    ax.set_ylabel("total steps per second")
    ax.set_title("Week 05: simulator throughput vs batch size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = "warp_throughput.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved plot -> {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
