# COMP0037 2026/27 — Coursework 2 Reference Solutions Execution Guide

**Instructor-only — do not distribute to students.**

This guide explains how to run, verify, and reproduce all reference solutions, diagnostic plots, and rollout GIFs for **Coursework 2 (MedBot Mobile Manipulator `PandaOmron` — Dual-Table Sample Transfer)** covering Questions 1–4 (Weeks 06–09).

All solution scripts dynamically monkey-patch the `TODO(CW2-...)` hooks in `../cw2_starter/` without modifying the student starter files and save all outputs to `assessment/CW02/cw2_solutions/outputs/`.

---

## 1. Environment Requirements

All CW2 scripts run in `.venv_robot` (matching Labs 06–09 in `TESTING_GUIDE.md`):

| Questions | Environment | Key Packages | Hardware |
|:---|:---|:---|:---|
| **Q1, Q2, Q3, Q4** | `.venv_robot` | `numpy<2`, `torch>=2.2`, `robosuite==1.5.2`, `mujoco==3.4.0`, `matplotlib`, `pillow` | CPU or CUDA GPU / Colab T4 |

### Windows PowerShell Session Variables (Recommended)
Run once per terminal session from the repository root (`0037_robotics_system-main/`):

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUTF8 = "1"
```

---

## 2. Running the Reference Solutions (Windows PowerShell)

### Question 1 — Mobile Manipulation Behavioral Cloning (BC)
Patches `train_bc` and `evaluate_bc_policy` in `cw2_starter/q1_mobile_lfd.py`.

```powershell
.venv_robot\Scripts\python assessment\CW02\cw2_solutions\q1_mobile_lfd_solutions.py --max-demos 16 --eval-episodes 4 --bc-epochs 120
```

- **Generated outputs:** `outputs/q1_lfd_comparison.png`, `outputs/q1_mobile_pick_place.gif` (side-by-side FSM Expert vs. BC rollout).
- **Verified sanity check:** 8-phase `MobilePickPlaceExpert` achieves **100% success** (mean `~156` steps); BC scales cleanly with dataset size while demonstrating mobile-manipulation covariate shift and out-of-distribution cube perturbation sensitivity.

---

### Question 2 — LLM Task Planning & Typed Mobile Skill API
Patches `validate_plan` and `MobileLLMPlanner` in `cw2_starter/q2_mobile_llm_planner.py`.

```powershell
.venv_robot\Scripts\python assessment\CW02\cw2_solutions\q2_mobile_llm_solutions.py
```

- **Generated outputs:** `outputs/q2_llm_plan_trace.png`, `outputs/q2_llm_rollout.gif` (83 frames).
- **Verified sanity check:** 5/5 schema & symbolic precondition tests `[PASS]`; closed-loop transfer succeeds in **1 attempt** (`drive A: 14`, `pick: 74`, `verify: 0`, `drive B: 31`, `place: 45` steps).

---

### Question 3 — Track A VLA (`NanoVLM` + Conditional Diffusion Chunk-MPC)
Patches `PixelShuffleProjector.forward`, `build_prefix_causal_mask`, `cosine_beta_schedule`, and `evaluate_chunk_mpc` in `cw2_starter/q3_mobile_vla.py`.

```powershell
.venv_robot\Scripts\python assessment\CW02\cw2_solutions\q3_mobile_vla_solutions.py --demos 10 --vlm-epochs 10 --diff-epochs 160 --eval-episodes 4
```

- **Generated outputs:** `outputs/q3_vla_training.png`, `outputs/q3_chunk_mpc.png`, `outputs/q3_vla_rollout.gif` (85 frames).
- **Verified sanity check (CPU, 10 demos, 10 VLM epochs, 160 DDPM epochs):**
  - `nanoVLM`: `Train CE = 0.3802`, `Val CE = 0.4769`, held-out VQA exact-match **`58.9%`** vs. majority-class baseline **`72.2%`**.
  - Closed-loop DDPM ablations:
    - `Proprio (30 ep, k=1)`: **0.0%** (`1.102 m` — undertraining lesson).
    - `Proprio (160 ep, k=1)`: **25.0%** (`0.835 m`).
    - **`Proprio + VLM (160 ep, k=1)`**: **100.0%** (`0.051 m`, `195.5` steps — frozen VLM visual embeddings dramatically improve closed-loop control).
    - `Proprio + VLM (160 ep, k=4)`: **75.0%** (`0.325 m`, `203.0` steps — 4× fewer diffusion calls with mild open-loop drift).

---

### Question 4 — Whole-Body CBF Safety Shield & 3-Scenario Red-Teaming
Patches `WholeBodyCBFShield.filter_action` and `run_red_team_suite` in `cw2_starter/q4_mobile_cbf_integration.py`.

```powershell
.venv_robot\Scripts\python assessment\CW02\cw2_solutions\q4_mobile_cbf_solutions.py
```

- **Generated outputs:** `outputs/q4_cbf_redteam.png`, `outputs/q4_cbf_nominal.gif`, `outputs/q4_cbf_adversarial.gif`, `outputs/q4_cbf_noisy.gif`.
- **Verified sanity check:**
  - `nominal`: Both Unshielded and CBF complete the full dual-table transfer (`succ=True`, `0` breaches, `59` CBF interventions).
  - `adversarial`: Unshielded suffers **76 base breaches & 58 EE breaches**; Whole-Body CBF logs **80 interventions** with **0 base breaches & 0 EE breaches**.
  - `noisy_perception`: Full `MobileSkillLibrary` transfer under depth-camera bias (`-12 cm` in $z$) + odometry bias (`-25 cm` in $x$) + noise; Unshielded suffers **208 base breaches & 19 EE breaches (`succ=False`)**, while Whole-Body CBF achieves **0 base breaches, 0 EE breaches, and `succ=True`** (`242` interventions)!

---

## 3. Regenerating the CW2 Figures & PDF

```powershell
.venv_robot\Scripts\python assessment\CW02\generate_cw2_figures.py
pdflatex --disable-installer -interaction=nonstopmode -output-directory=assessment/CW02 assessment/CW02/COMP0037_2627_CW2.tex
pdflatex --disable-installer -interaction=nonstopmode -output-directory=assessment/CW02 assessment/CW02/COMP0037_2627_CW2.tex
```
