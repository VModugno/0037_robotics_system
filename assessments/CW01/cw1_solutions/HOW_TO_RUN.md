# COMP0037 2026/27 — Coursework 1 Reference Solutions Execution Guide

**Instructor-only — do not distribute to students.**

This guide provides complete, step-by-step instructions to run, verify, and reproduce all reference code solutions and figures for **Coursework 1 (MedBot: Autonomous Hospital Pharmacy & Sterile Laboratory Assistant)** covering Questions 1–4 (Weeks 01–05).

All solution scripts dynamically monkey-patch the `TODO(CW1-...)` hooks in `../cw1_starter/` without modifying the student starter files, print quantitative summary tables to stdout, and save all diagnostic figures into `assessment/cw1_solutions/outputs/`.

---

## 1. Environment Requirements

Two virtual environments (matching the repository's local setup in `TESTING_GUIDE.md`) are used:

| Questions | Environment | Key Packages | Hardware |
|:---|:---|:---|:---|
| **Q1, Q2, Q3** | `.venv_rl` | `numpy>=1.26`, `matplotlib>=3.8`, `torch>=2.2` | CPU (GPU optional for Q3) |
| **Q4** | `.venv_sim` | `numpy>=1.26`, `matplotlib>=3.8`, `mujoco>=3.13`, `mujoco-warp>=4.0`, `warp-lang>=1.15` | CPU + CUDA GPU (or Colab T4) |

### Windows PowerShell Session Variables (Recommended)
Run once per terminal session from the repository root (`0037_robotics_system-main/`):

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUTF8 = "1"
```

*(On Linux / macOS, use `export KMP_DUPLICATE_LIB_OK=TRUE` and `export PYTHONUTF8=1`.)*

---

## 2. Running the Reference Solutions (Local — Windows PowerShell)

All commands below assume you are in the **repository root** (`0037_robotics_system-main/`).

### Question 1 — Hospital Dispatch Bay Bandit (Stationary & Non-Stationary)
Patches `run_ucb1` and `run_epsilon_greedy_const_alpha` in `cw1_starter/q1_bandits_hospital.py`.

```powershell
.venv_rl\Scripts\python assessment\cw1_solutions\q1_bandits_solutions.py
```

- **Optional flags:** `--runs 20` (default), `--out-dir assessment\cw1_solutions\outputs`
- **Runtime:** ~2–3 seconds on CPU.
- **Generated figures:**
  - `outputs/q1_stationary.png` — Cumulative regret and % optimal bay over $T = 5000$ (20 runs).
  - `outputs/q1_drifting.png` — Sample-average $\varepsilon$-greedy vs. constant-$\alpha$ $\varepsilon$-greedy vs. UCB1 under `drift_std = 0.05` ($T = 2000$).
- **Reference sanity-check numbers (20 runs):**
  - Stationary ($T = 5000$): `UCB1 c=1.0` achieves **~61.9** cumulative regret and **~94.9%** optimal actions (vs. greedy `~1818.6` / `50.0%`).
  - Drifting (`drift_std = 0.05`, $T = 2000$): `Const-alpha (a=0.1) eps=0.1` tracks the optimal bay at **~72.1%** (last 200 steps) vs. Sample-avg **~58.1%** and UCB1 **~49.1%**.

---

### Question 2 — Hospital Ward Stochastic Navigation (`HospitalWardEnv`)
Patches `train_tabular` in `cw1_starter/q2_hospital_ward.py` (8×12 grid, `start=(7, 0)`, `goal=(0, 11)`).

```powershell
.venv_rl\Scripts\python -X utf8 assessment\cw1_solutions\q2_hospital_ward_solutions.py
```

- **Optional flags:** `--episodes 800` (default), `--out-dir assessment\cw1_solutions\outputs`
- **Runtime:** ~3–5 seconds on CPU.
- **Note:** `-X utf8` ensures Unicode policy arrows (`↑ → ↓ ←`) render cleanly in Windows PowerShell.
- **Generated figures:**
  - `outputs/q2_slip_sweep.png` — 2×2 learning curves for Q-Learning vs. SARSA across $p_{\text{slip}} \in \{0.0, 0.1, 0.2, 0.4\}$ ($\varepsilon = 0.1$).
  - `outputs/q2_eps_sweep.png` — 1×3 learning curves across $\varepsilon \in \{0.05, 0.1, 0.25\}$ ($p_{\text{slip}} = 0.2$).
- **Reference sanity-check numbers (800 episodes, last-100 mean return):**
  - At `slip = 0.2, eps = 0.1`: **SARSA** learns the conservative far-left corridor away from the biohazard spill (`+7.81`) vs. **Q-Learning** (`+5.05`).

---

### Question 3 — Sterile Cabinet 2-DoF Arm (DQN/Double DQN & Dual-Barrier CBF A2C)
Runs discrete DQN variants (`Standard DQN`, `Double DQN`, `No Target Net`) on `SterileArmEnv(mode="discrete")`, then patches `cbf_safety_shield` in `cw1_starter/q3_sterile_arm.py` and compares `Unshielded A2C`, `Soft-Penalty A2C`, and `Dual-Barrier CBF A2C` on `SterileArmEnv(mode="continuous")`.

**Quick smoke test (1 seed, 80 episodes — ~15 seconds on CPU):**
```powershell
.venv_rl\Scripts\python assessment\cw1_solutions\q3_sterile_arm_solutions.py --fast
```

**Full coursework configuration (3 seeds, 300 DQN / 500 A2C episodes — ~2–3 minutes on CPU):**
```powershell
.venv_rl\Scripts\python assessment\cw1_solutions\q3_sterile_arm_solutions.py --dqn-episodes 300 --a2c-episodes 500 --seeds 3
```

- **Optional GIF seed flag:** `--gif-seed -1` (default) automatically selects a contrasting test episode where `Unshielded A2C` hits the centrifuge while `Dual-Barrier CBF A2C` avoids it cleanly; pass an explicit integer (e.g., `--gif-seed 8`) to inspect a specific episode.
- **Generated figures & animation:**
  - `outputs/q3_dqn_comparison.png` — Moving-average returns for Standard DQN, Double DQN, and No-Target-Net.
  - `outputs/q3_a2c_safety.png` — 3-panel comparison: Episode Return, Cumulative Joint-Limit Violations, and Cumulative Centrifuge Collisions.
  - `outputs/q3_sterile_arm_rollout.gif` — Synchronized 3-panel rollout animation (Double DQN vs. Unshielded A2C vs. Dual-Barrier CBF A2C) showing end-effector trajectory, centrifuge obstacle avoidance, and live safety counters.
- **Reference sanity-check numbers (1 seed, 250 DQN / 500 A2C episodes):**
  - Discrete DQN: `Standard DQN` **100.0%** (`0.037 m`, chatter `2.39 rad`), `Double DQN` **100.0%** (`0.033 m`, chatter `2.23 rad`), `No Target Net` **0.0%** (Deadly Triad instability).
  - Continuous A2C:
    - `Unshielded A2C`: Return `+2.7`, **2,589** joint-limit hits, **758** centrifuge collisions, **100.0%** greedy success.
    - `Soft-Penalty A2C`: Return `+1.3`, **3,363** joint-limit hits, **64** centrifuge collisions, **95.0%** greedy success (soft penalties cannot prevent exploration collisions).
    - `Dual-Barrier CBF A2C`: Return **`+2.9`**, **0.0** joint-limit hits, **0.0** centrifuge collisions, **95.0%** greedy success.

---

### Question 4 — MedBot Payload Trolley (CPU vs. GPU MuJoCo Warp & 3-Param DR)
Patches `sample_trolley_randomization` in `cw1_starter/q4_payload_sim.py` and benchmarks `medbot_payload_trolley` across CPU MuJoCo and CUDA-Graph-captured GPU MuJoCo Warp, then runs 3-parameter Domain Randomization (`mast_damping`, `floor_friction`, `payload_mass`).

```powershell
$env:WARP_CACHE_PATH = (Join-Path (Get-Location) ".warp_cache")
.venv_sim\Scripts\python assessment\cw1_solutions\q4_payload_sim_solutions.py
```

- **Fast smoke-test flags:** `--nworlds 1,4,16,64,256 --bench-steps 200 --dr-nworld 32 --dr-steps 500`
- **Runtime:** ~15–25 seconds with cached Warp kernels (first-ever JIT compilation on a new GPU takes ~30–60 seconds).
- **Generated figures:**
  - `outputs/q4_warp_throughput.png` — Log-log total steps/s (with CPU baseline dashed line) and per-world steps/s.
  - `outputs/q4_trolley_dr.png` — 3-panel mast tilt trajectory fan-out ($N = 32$ worlds, 500 steps from $\theta_0 = +0.25\text{ rad}$): Narrow (log-uniform $d$), Wide (log-uniform $d$), and Wide (linear-uniform $d$).
- **Reference sanity-check numbers (RTX 4050 Laptop GPU, `cuda:0`):**
  - CPU baseline: **~491,500 steps/s**
  - Warp GPU: **~3,800 steps/s** (`nworld=1`) → **~282,000 steps/s** (`nworld=64`) → **~1,065,000 steps/s** (`nworld=256`, crossover around `nworld ≈ 110–130`).
  - DR spread ($N = 32$, 500 steps): Wide log-uniform $\sigma_{\text{final}} \approx \mathbf{1.03\text{ rad}}$ vs. Wide linear-uniform $\sigma_{\text{final}} \approx \mathbf{0.63\text{ rad}}$ (linear sampling under-samples the low-damping underdamped regime).

---

## 3. Running All Four Solutions on Google Colab (T4 GPU)

If you want to verify the entire CW1 solution suite on **Google Colab** (Runtime $\to$ Change runtime type $\to$ **T4 GPU**):

```bash
# 1. Install dependencies (Weeks 01-05 stack)
!pip install -q "numpy>=1.26" "matplotlib>=3.8" "torch>=2.2" "mujoco>=3.13" "mujoco-warp>=4.0" warp-lang

# 2. Run Q1, Q2, Q3, and Q4 solutions from the repository root
!python assessment/cw1_solutions/q1_bandits_solutions.py
!python -X utf8 assessment/cw1_solutions/q2_hospital_ward_solutions.py
!python assessment/cw1_solutions/q3_sterile_arm_solutions.py --fast
!python assessment/cw1_solutions/q4_payload_sim_solutions.py
```

Display any generated plot inline:
```python
from IPython.display import Image, display
display(Image(filename="assessment/cw1_solutions/outputs/q3_a2c_safety.png"))
display(Image(filename="assessment/cw1_solutions/outputs/q4_trolley_dr.png"))
```

---

## 4. Troubleshooting

| Issue | Cause | Fix |
|:---|:---|:---|
| `UnicodeEncodeError: 'charmap' codec can't encode character '\u2191'` | Windows PowerShell cp1252 console encoding in Q2 | Pass `-X utf8` to Python or set `$env:PYTHONUTF8 = "1"`. |
| `OMP: Error #15: Initializing libiomp5md.dll` | OpenMP conflict between PyTorch and NumPy/MKL on Windows | Set `$env:KMP_DUPLICATE_LIB_OK = "TRUE"` before running Q3. |
| `PermissionError` or slow JIT compile in Q4 (`warp/Cache`) | Default Warp cache path blocked or uncached | Set `$env:WARP_CACHE_PATH = "$PWD\.warp_cache"` (Windows) or `export WARP_CACHE_PATH=./.warp_cache` (Linux). |
| `OSError: libmujoco.so.x: cannot open shared object file` on Colab/Linux | Linux dynamic loader cannot resolve MuJoCo bundled plugins | Already handled automatically by `trolley_model()` in `cw1_starter/q4_payload_sim.py` via `ctypes.RTLD_GLOBAL`. |
| `wp.get_device()` reports `cpu` on Colab | Colab T4 driver (CUDA 12.4) vs. CUDA 13 wheel | Install the `+cu12` wheel or pin `warp-lang==1.18.0`, then restart the Colab runtime. |

