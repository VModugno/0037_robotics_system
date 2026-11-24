# COMP0037 2026/27 — Coursework 2 Reference Solutions (`cw2_solutions/`)

**Instructor-only — do not distribute to students.**

These scripts dynamically patch the `TODO(CW2-...)` hooks in `assessment/CW02/cw2_starter/` without modifying the starter files, run all required Coursework 2 experiments on `MobilePickPlaceEnv` (`PandaOmron`), print quantitative summary tables, and save diagnostic plots and rollout GIFs under `assessment/CW02/cw2_solutions/outputs/`.

## Running the Reference Solutions (`.venv_robot`)

From the repository root:

```powershell
# Q1 (Behavioral Cloning: 8-phase PandaOmron FSM Expert, BC dataset scaling, Noise-Aug BC & perturbation)
.venv_robot\Scripts\python assessment\CW02\cw2_solutions\q1_mobile_lfd_solutions.py

# Q2 (Typed Mobile-Manipulation Skill API, schema/precondition validator & LLM planner)
.venv_robot\Scripts\python assessment\CW02\cw2_solutions\q2_mobile_llm_solutions.py

# Q3 (Track A VLA: nanoVLM from scratch + Conditional Diffusion Action Head + Chunk-MPC)
.venv_robot\Scripts\python assessment\CW02\cw2_solutions\q3_mobile_vla_solutions.py

# Q4 (4-tier Whole-Body CBF Safety Shield + 3-scenario Red-Teaming)
.venv_robot\Scripts\python assessment\CW02\cw2_solutions\q4_mobile_cbf_solutions.py
```

## Generated Reference Artifacts (`outputs/`)
- `q1_lfd_comparison.png`, `q1_mobile_pick_place.gif`
- `q2_llm_plan_trace.png`, `q2_llm_rollout.gif`
- `q3_vla_training.png`, `q3_chunk_mpc.png`, `q3_vla_rollout.gif`
- `q4_cbf_redteam.png`, `q4_cbf_nominal.gif`, `q4_cbf_adversarial.gif`, `q4_cbf_noisy.gif`

