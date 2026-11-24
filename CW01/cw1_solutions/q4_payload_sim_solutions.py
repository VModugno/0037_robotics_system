"""Instructor reference solutions for COMP0037 2026/27 Coursework 1 — Question 4.

Patches TODO(CW1-Q4c) (`sample_trolley_randomization`) in
`assessment/cw1_starter/q4_payload_sim.py` and runs:
  - Q4.b: CPU MuJoCo vs. GPU MuJoCo Warp throughput sweep (nworlds = 1..256)
  - Q4.c/d: 3-parameter Domain Randomization on `medbot_payload_trolley`
            comparing narrow vs. wide ranges and linear vs. log-uniform damping.

Usage (from repository root, inside .venv_sim):
    .venv_sim\\Scripts\\python assessment\\cw1_solutions\\q4_payload_sim_solutions.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
STARTER_DIR = os.path.normpath(os.path.join(CUR_DIR, os.pardir, "cw1_starter"))
OUT_DIR = os.path.join(CUR_DIR, "outputs")
sys.path.insert(0, STARTER_DIR)

import q4_payload_sim as q4  # noqa: E402

MAST_DOF = 1
FLOOR_GEOM = 0
PAYLOAD_BODY = 3


# --------------------------------------------------------------------------- #
# Reference implementation for Q4 TODO
# --------------------------------------------------------------------------- #

def sample_trolley_randomization(
    rng: np.random.Generator,
    nworld: int,
    log_uniform_damping: bool = True,
    damping_range: tuple[float, float] = (1e-3, 1.0),
    friction_range: tuple[float, float] = (0.2, 1.5),
    mass_range: tuple[float, float] = (0.2, 1.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample per-world (mast_damping, floor_friction, payload_mass)."""
    d_lo, d_hi = damping_range
    if log_uniform_damping:
        mast_damping = np.exp(rng.uniform(np.log(d_lo), np.log(d_hi), nworld)).astype(np.float32)
    else:
        mast_damping = rng.uniform(d_lo, d_hi, nworld).astype(np.float32)
    floor_friction = rng.uniform(friction_range[0], friction_range[1], nworld).astype(np.float32)
    payload_mass = rng.uniform(mass_range[0], mass_range[1], nworld).astype(np.float32)
    return mast_damping, floor_friction, payload_mass


def patch_all() -> None:
    q4.sample_trolley_randomization = sample_trolley_randomization


def benchmark_cpu(mjm, steps: int = 5000, repeats: int = 5) -> float:
    import mujoco

    best = float("inf")
    for _ in range(repeats):
        mjd = mujoco.MjData(mjm)
        mjd.qpos[MAST_DOF] = 0.25
        t0 = time.perf_counter()
        for _ in range(steps):
            mujoco.mj_step(mjm, mjd)
        best = min(best, time.perf_counter() - t0)
    return steps / best


def benchmark_warp(mjm, nworlds: list[int], steps: int = 500) -> dict[int, float]:
    import mujoco_warp as mjw
    import warp as wp

    results = {}
    for nw in nworlds:
        model = mjw.put_model(mjm)
        data = mjw.make_data(mjm, nworld=nw)
        qpos = np.zeros((nw, mjm.nq), dtype=np.float32)
        qpos[:, MAST_DOF] = 0.25
        data.qpos = wp.array(qpos, dtype=wp.float32)
        mjw.forward(model, data)

        # Warm-up JIT compilation
        for _ in range(5):
            mjw.step(model, data)
        wp.synchronize()

        # Capture a CUDA graph on GPU to eliminate Python per-step dispatch overhead
        if str(wp.get_device()).startswith("cuda"):
            with wp.ScopedCapture() as cap:
                mjw.step(model, data)
            wp.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                wp.capture_launch(cap.graph)
            wp.synchronize()
        else:
            t0 = time.perf_counter()
            for _ in range(steps):
                mjw.step(model, data)
            wp.synchronize()
        dt = time.perf_counter() - t0
        results[nw] = (nw * steps) / dt
    return results


def run_trolley_dr(
    mjm,
    nworld: int,
    steps: int,
    log_uniform_damping: bool,
    damping_range: tuple[float, float],
    friction_range: tuple[float, float],
    mass_range: tuple[float, float],
    seed: int = 0,
) -> np.ndarray:
    import mujoco_warp as mjw
    import warp as wp

    rng = np.random.default_rng(seed)
    damps, frics, masses = sample_trolley_randomization(
        rng,
        nworld=nworld,
        log_uniform_damping=log_uniform_damping,
        damping_range=damping_range,
        friction_range=friction_range,
        mass_range=mass_range,
    )

    model = mjw.put_model(mjm)
    d_arr = np.repeat(model.dof_damping.numpy(), nworld, axis=0)
    d_arr[:, MAST_DOF] = damps
    model.dof_damping = wp.array(d_arr, dtype=wp.float32)

    f_arr = np.repeat(model.geom_friction.numpy(), nworld, axis=0)
    f_arr[:, FLOOR_GEOM, 0] = frics
    model.geom_friction = wp.array(f_arr, dtype=wp.vec3f)

    m_arr = np.repeat(model.body_mass.numpy(), nworld, axis=0)
    m_arr[:, PAYLOAD_BODY] = masses
    model.body_mass = wp.array(m_arr, dtype=wp.float32)

    data = mjw.make_data(mjm, nworld=nworld)
    qpos = np.zeros((nworld, mjm.nq), dtype=np.float32)
    qpos[:, MAST_DOF] = 0.25
    data.qpos = wp.array(qpos, dtype=wp.float32)
    mjw.forward(model, data)

    traj = [data.qpos[:, MAST_DOF].numpy().copy()]
    for i in range(steps):
        mjw.step(model, data)
        if (i + 1) % 10 == 0:
            traj.append(data.qpos[:, MAST_DOF].numpy().copy())
    return np.asarray(traj)


def main() -> None:
    ap = argparse.ArgumentParser(description="CW1 Q4 reference solutions runner")
    ap.add_argument("--nworlds", type=str, default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--bench-steps", type=int, default=500)
    ap.add_argument("--dr-nworld", type=int, default=32)
    ap.add_argument("--dr-steps", type=int, default=500)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    patch_all()
    os.makedirs(args.out_dir, exist_ok=True)
    mjm = q4.trolley_model()

    # ── Q4.b: CPU vs MuJoCo Warp throughput ───────────────────────────── #
    import warp as wp
    print("=" * 72)
    print(f"CW1 Q4.b — MedBot Trolley CPU vs. MuJoCo Warp ({wp.get_device()})")
    print("=" * 72)
    cpu_sps = benchmark_cpu(mjm, steps=5000, repeats=5)
    print(f"  CPU baseline (best of 5, 5000 steps): {cpu_sps:>12,.0f} steps/s")

    nworlds = [int(x) for x in args.nworlds.split(",")]
    warp_res = benchmark_warp(mjm, nworlds=nworlds, steps=args.bench_steps)
    for nw in nworlds:
        print(f"  Warp {nw:>4} worlds: {warp_res[nw]:>12,.0f} total steps/s  ({warp_res[nw]/nw:>10,.0f} /world)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ns = sorted(warp_res)
    tot = [warp_res[n] for n in ns]
    pw = [warp_res[n] / n for n in ns]
    ax1.loglog(ns, tot, "o-", lw=1.8, label="MuJoCo Warp (total)")
    ax1.axhline(cpu_sps, color="#DC2626", ls="--", lw=1.5, label=f"CPU baseline ({cpu_sps:,.0f}/s)")
    ax1.set(xlabel="nworld", ylabel="Total steps/s", title="Q4.b Total Throughput")
    ax2.loglog(ns, pw, "s-", lw=1.8, color="#2563EB", label="Per-world steps/s")
    ax2.set(xlabel="nworld", ylabel="Steps/s per world", title="Q4.b Per-World Throughput")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, "q4_warp_throughput.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p1}\n")

    # ── Q4.c/d: 3-Parameter Domain Randomization comparison ───────────── #
    print("=" * 72)
    print(f"CW1 Q4.c/d — 3-Param Trolley DR (N={args.dr_nworld} worlds, {args.dr_steps} steps)")
    print("=" * 72)
    dr_configs = [
        ("Narrow (log-uniform d)", True, (0.02, 0.15), (0.6, 1.0), (0.5, 0.8)),
        ("Wide (log-uniform d)", True, (1e-3, 1.0), (0.2, 1.5), (0.2, 1.5)),
        ("Wide (linear-uniform d)", False, (1e-3, 1.0), (0.2, 1.5), (0.2, 1.5)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)
    t_axis = np.arange(args.dr_steps // 10 + 1) * 10 * 0.01
    for ax, (label, is_log, dr_d, dr_f, dr_m) in zip(axes, dr_configs):
        traj = run_trolley_dr(mjm, args.dr_nworld, args.dr_steps, is_log, dr_d, dr_f, dr_m, seed=0)
        final = traj[-1]
        print(f"  {label:<25} | start std={traj[0].std():.4f} | final mean={final.mean():+.3f} rad, sigma_final={final.std():.4f} rad")
        for w in range(args.dr_nworld):
            ax.plot(t_axis, traj[:, w], lw=0.8, alpha=0.65)
        ax.set(xlabel="Time (s)", title=f"{label}\n(sigma_final = {final.std():.3f} rad)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mast tilt angle (rad)")
    fig.tight_layout()
    p2 = os.path.join(args.out_dir, "q4_trolley_dr.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p2}")
    print("CW1_Q4_SOLUTIONS_DONE")


if __name__ == "__main__":
    main()

