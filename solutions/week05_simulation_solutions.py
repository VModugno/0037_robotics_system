"""
Instructor solutions for COMP0037 Week 05 (week05_simulation).

Reference implementations for the six student TODOs are monkey-patched onto
the lab modules, then all three mains run:

  T1  benchmark_mujoco_cpu.time_cpu_rollout  -> module-attribute assignment
  T2  benchmark_mujoco_warp.make_batched     -> module-attribute assignment
  T3  benchmark_mujoco_warp.time_batched_steps (full reference: CUDA-graph
      capture on CUDA devices, plain loop on CPU)
  T4  benchmark_mujoco_warp.check_overflow  -> module-attribute assignment
  T5  domain_randomization.sample_randomization
  T6  domain_randomization.run_randomized_rollout

INSTRUCTOR USE ONLY -- students implement the TODOs themselves.
Part A (CPU baseline) runs anywhere; parts B/C need the warp runtime, which
on this instructor Windows machine is blocked by App Control policy
(WinError 4551) -- the scripts exit gracefully there; run them on Colab
(warp_throughput.png / domain_randomization.png land in the cwd).

Reference number from this machine: CPU cartpole baseline ~2.1e5 steps/s
(best-of-5 over 5000 steps) -- matches the LAB.md self-check.

Run from the labs/ root:
    pixi run --manifest-path pixi.toml -e week05 python solutions/week05_simulation_solutions.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

LAB_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "week05_simulation")
)
sys.path.insert(0, LAB_DIR)

print("=" * 70)
print("PART A: CPU baseline")
print("=" * 70)
import benchmark_mujoco_cpu as bcpu


def time_cpu_rollout(m, n_steps, n_repeats=5):
    bcpu.rollout_cpu(m, n_steps)  # warm-up (JIT/alloc effects)
    best = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        bcpu.rollout_cpu(m, n_steps)
        best = min(best, time.perf_counter() - t0)
    return n_steps / best


bcpu.time_cpu_rollout = time_cpu_rollout
sys.argv = ["bench_cpu", "--steps", "5000"]
bcpu.main()

print()
print("=" * 70)
print("PART B: MuJoCo Warp batched")
print("=" * 70)
import warp as wp
import mujoco_warp as mjw
import benchmark_mujoco_warp as bwarp


def make_batched(mjm, nworld, seed=0):
    model = mjw.put_model(mjm)
    data = mjw.make_data(mjm, nworld=nworld)
    qvel = data.qvel.numpy()
    rng = np.random.default_rng(seed)
    qvel[:, 1] = rng.uniform(-1.0, 1.0, size=nworld)
    data.qvel = wp.array(qvel, dtype=wp.float32)
    mjw.forward(model, data)
    return model, data


def time_batched_steps(model, data, n_steps):
    mjw.step(model, data)  # warm-up: JIT-compiled kernels, do NOT time
    wp.synchronize()
    nworld = data.qpos.shape[0]
    if wp.get_device().is_cuda:
        with wp.ScopedCapture() as capture:
            mjw.step(model, data)
        wp.capture_launch(capture.graph)  # finish compiling the graph
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            wp.capture_launch(capture.graph)
        wp.synchronize()  # wait for the last launch before stopping the clock
        dt = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        for _ in range(n_steps):
            mjw.step(model, data)
        wp.synchronize()
        dt = time.perf_counter() - t0
    return n_steps * nworld / dt


def check_overflow(data):
    ov = data.overflow.numpy()
    bad = np.nonzero(ov)[0]
    return bad.astype(int) if len(bad) else np.empty(0, dtype=int)


bwarp.make_batched = make_batched
bwarp.time_batched_steps = time_batched_steps
bwarp.check_overflow = check_overflow
sys.argv = ["bench_warp", "--nworlds", "1,4,16", "--steps", "500"]
try:
    bwarp.main()
except SystemExit as e:
    print("SYSEXIT:", e)

print()
print("=" * 70)
print("PART C: domain randomization")
print("=" * 70)
import domain_randomization as dr


def sample_randomization(rng, nworld):
    damp = np.exp(rng.uniform(np.log(1e-3), np.log(1.0), nworld)).astype(np.float32)
    fric = rng.uniform(0.2, 1.5, nworld).astype(np.float32)
    return damp, fric


def run_randomized_rollout(model, data, n_steps, sample_every=10):
    out = []
    for i in range(n_steps):
        mjw.step(model, data)
        if (i + 1) % sample_every == 0:
            out.append(data.qpos[:, 1].numpy().copy())
    return out


dr.sample_randomization = sample_randomization
dr.run_randomized_rollout = run_randomized_rollout
sys.argv = ["dr", "--nworld", "8", "--steps", "300"]
try:
    dr.main()
except SystemExit as e:
    print("SYSEXIT:", e)

print()
print("WEEK05_SOLUTIONS_DONE")
