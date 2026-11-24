# Week 05 — Simulation at Scale: MuJoCo CPU vs MuJoCo Warp

## 1. Overview

Modern robot learning needs *millions* of environment steps, and no single
CPU core can provide them. This lab has you benchmark two faces of the same
physics engine:

- **Plain MuJoCo (CPU)** — the reference single-environment simulator,
  ~10⁵ steps/s on one core for a small model.
- **MuJoCo Warp (`mujoco-warp`)** — the same physics re-implemented as
  NVIDIA Warp GPU kernels. One `mjw.step` call advances **`nworld`**
  independent worlds at once on the GPU.

You will also use Warp's **batched model fields** to do *domain
randomization* (DR): every world gets its own damping and friction, which is
how sim-to-real robustness is trained.

**Learning outcomes.** After this lab you can:

1. Benchmark a simulator honestly (warm-up, CUDA-graph capture, synchronize).
2. Explain latency vs throughput: why 1 GPU world loses to CPU but 1000 win.
3. Read the MuJoCo Warp data model (`put_model`, `make_data`, `Data.qpos`).
4. Set up domain randomization over batched physics parameters.
5. Argue when to pick MuJoCo / MJX / MuJoCo Warp (see discussion Q6).

## 2. Setup

Local (from the `labs/` folder):

```
pixi run -e week05 python week05_simulation/sim_models.py                 # self-test
pixi run -e week05 python week05_simulation/benchmark_mujoco_cpu.py
pixi run -e week05 python week05_simulation/benchmark_mujoco_warp.py
pixi run -e week05 python week05_simulation/domain_randomization.py
```

The week05 pixi environment ships `mujoco`, `mujoco-warp` and `warp-lang`.
On a **GPU machine** Warp picks `cuda:0` automatically. On CPU-only machines
Warp runs on the `cpu` device (slower — that is a lab observation, not a bug).

### Colab (T4 GPU)

```python
!pip install -q numpy matplotlib "mujoco>=3.13" "mujoco-warp>=4.0"
# The newest warp-lang wheel is built against CUDA 13.x and needs driver R580+.
# Colab T4s still run CUDA 12.4 (driver 550.x) -> install the +cu12 wheel:
!pip install -q https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl
# (If that URL ever 404s, fall back to:  !pip install -q "warp-lang==1.18.0")

import warp as wp
wp.init()
print(wp.get_device())   # must print cuda:0 - if it says cpu, the GPU path below is invalid
```

Then upload the `week05_simulation/` folder and run the three scripts.
**Every timing number in your lab notes must state the device it ran on.**

## 3. Your tasks (TODO map)

| Task | File | Function | What you implement |
|------|------|----------|---------------------|
| T1 | `benchmark_mujoco_cpu.py` | `time_cpu_rollout` | warm-up + `perf_counter` timing, best of N |
| T2 | `benchmark_mujoco_warp.py` | `make_batched` | `put_model` + `make_data(nworld=...)` + per-world init |
| T3 | `benchmark_mujoco_warp.py` | `time_batched_steps` | warm-up, CUDA-graph capture, synchronized timing |
| T4 | `benchmark_mujoco_warp.py` | `check_overflow` | read `Data.overflow` bitmask, return offending worlds |
| T5 | `domain_randomization.py` | `sample_randomization` | log-uniform damping, uniform friction per world |
| T6 | `domain_randomization.py` | `run_randomized_rollout` | step all worlds, record pole angles |

Read `sim_models.py` first — the embedded cartpole MJCF is the common
substrate for all three parts (2 DoF: slide + hinge, 1 actuator).

Key API facts you will need (verify against the installed package with
`inspect.signature` if unsure):

```python
import mujoco_warp as mjw
m = mjw.put_model(mjm, batch_sizes={"dof_damping": 8})   # upload model (optionally batch some fields)
d = mjw.make_data(mjm, nworld=8)                         # batched data, qpos shape (8, nq)
mjw.step(m, d)                                           # advance ALL worlds
d.qpos.numpy()                                           # copy back to host numpy
```

## 4. Experiments

1. **Part A — CPU baseline.** Run `benchmark_mujoco_cpu.py` with the default
   `--steps 5000 --repeats 5`. Record steps/s. Re-run with `--steps 20000`
   and note whether the number moves (why might it?).
2. **Part B — batched throughput.** Run the Warp benchmark with the default
   world sweep `1,2,4,8,16,32,64`. Pass your Part A number via
   `--cpu-baseline`. Save `warp_throughput.png`.
3. **Part C — domain randomization.** Run with `--nworld 16 --steps 500`.
   Save `domain_randomization.png` and record start/end pole-angle std.
4. **CPU-device observation.** On a CPU-only machine, run Part B anyway and
   compare against Part A — batching overhead without parallelism.
5. **(GPU only) Graph capture on/off.** Modify your T3 to time *without*
   CUDA-graph capture (plain `mjw.step` loop) and compare.

## 5. Self-check

Reference run (instructor laptop, CPU device, CPU benchmark only —
GPU numbers must come from your own Colab run):

```
MuJoCo CPU baseline (single environment, 5000 steps/rollout)
  best of 5: 216,034 steps/s
```

Part A on a modern laptop lands in the 1.5–4 × 10⁵ steps/s range for this
model — the exact number depends on CPU, BLAS and Python version, so **your
number is the reference for your report**, not mine. If Part A prints
< 30,000 steps/s, check that you did not leave a debugger attached.

Parts B/C sanity signals:

- Cartpole has few contacts, so `check_overflow` should always return an
  empty array; any nonzero bit means your buffers are undersized.
- All worlds share the same start state (T2 only perturbs `qvel[:, 1]`),
  so before the first step the pole angles must be identical.
- DR spread: at 500 steps the final pole-angle std should be O(0.1–1 rad)
  with the default ranges — if it is ~0, your T5 sampling never varied the
  hinge damping (check that `dampings[:, HINGE_DOF]` actually changes).

## 6. Deliverables & Discussion Questions

Record your experimental findings and answer the following questions in your lab notes:

**Deliverables checklist**

- [ ] `warp_throughput.png` (Part B, GPU) with the CPU baseline line drawn
- [ ] `domain_randomization.png` (Part C)
- [ ] A table: `nworld | total steps/s | steps/s per world` for your run
- [ ] Your Part A number + CPU model string, and your GPU number + GPU model
- [ ] Answers to Q1–Q6 below with the numbers you actually measured

**Questions**

- **Q1 (latency vs throughput).** Using your table, compute the smallest
  `nworld` at which Warp beats your CPU baseline. Explain why `nworld=1` on
  GPU loses to a single CPU core (think: kernel launch + host↔device traffic
  per step, versus one core running tight C code).
- **Q2 (per-world decay).** Plot/inspect `steps/s per world` vs `nworld`.
  Does per-world throughput fall, rise, or stay flat? What saturates first
  on your device: SM occupancy, or Python dispatch overhead?
- **Q3 (graph capture).** Compare timing with and without CUDA-graph capture
  (experiment 5). One `mjw.step` is hundreds of kernel launches — quantify
  what fraction of step time is launch overhead at `nworld=1` vs `nworld=64`.
- **Q4 (fixed buffers).** `make_data` allocates fixed-size contact buffers
  (`nconmax`, `njmax`). Explain what `Data.overflow` would tell you, and why
  MuJoCo Warp prefers fixed-size buffers over dynamic allocation on GPU.
- **Q5 (DR spread).** Compare start vs final pole-angle std from Part C.
  How would you turn the measured spread into a *training* distribution for
  a policy (what to randomize, how wide, and why log-uniform damping)?
- **Q6 (engine choice).** MJX (JAX) is differentiable and TPU-friendly;
  MuJoCo Warp is faster on complex scenes and not differentiable. Argue for
  one engine for (a) a course project doing RL on a 7-DoF arm, (b) a
  research project needing gradients through contact, (c) a T4 Colab budget.

**Reproducibility block** — state for every number: machine (CPU/GPU model),
device (`cpu`/`cuda:0`), command line incl. flags, and the seed. The scripts
default to seed 0 for T2/T5 sampling; timing runs are inherently noisy, so
record best-of-5 (Part A) and single-run (Part B) numbers with that caveat.

## 7. Going further

- Benchmark a **humanoid**: swap in a model from
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  (e.g. `humanoid`) and rerun Part B. Complex scenes are exactly where
  MuJoCo Warp claims its biggest wins over MJX — test that claim.
- **Determinism**: MuJoCo Warp is non-deterministic by default on GPU
  (atomics). Try `m.opt.deterministic = mjw.DeterminismType.ALL` and
  measure the throughput cost.
- **Buffer sizing**: shrink `nconmax` in `make_data` until `check_overflow`
  fires on the humanoid; report the tightest safe value you find.
- MJX: if you finish early, replicate Part B with `mjx` on the same model
  and add the curve to your plot (this is the real Weeks 5 comparison from
  the research literature: MuJoCo Warp vs MJX, >100× on complex scenes).

## 8. References

- MuJoCo Warp docs — https://mujoco.readthedocs.io/en/stable/mjwarp/
- MuJoCo Warp repo — https://github.com/google-deepmind/mujoco_warp
- "When to use MuJoCo Warp" (MJX comparison) — docs §When to use
- NVIDIA Warp — https://github.com/NVIDIA/warp
- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks
  from Simulation to the Real World*, IROS 2017.
- Peng, Coumans et al., *Advancing Physics Simulation for Robotics*
  (Newton / MuJoCo Warp), 2025.

## Troubleshooting

- **`WinError 4551` / "Application Control policy has blocked this file"**
  (Windows): your machine's security policy blocks `warp\bin\warp-clang.dll`.
  The scripts print a readable error and exit; run the GPU parts on Colab.
- **`wp.get_device()` says `cpu` on Colab**: the CUDA wheel did not match the
  driver — redo the `+cu12` wheel install from §2.
- **First Part B call takes seconds**: that is JIT compilation, not a bug —
  your T3 warm-up exists precisely to keep it out of the measurement.
