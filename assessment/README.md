# COMP0037 (AY 2026/27) — Assessment Package Guide

> **Audience:** COMP0037 Teaching Team (Instructor-Only Overview)  
> **Important:** Do **not** distribute `cw1_solutions/` or `cw2_solutions/` to students. Ship only the compiled coursework PDFs (`COMP0037_2627_CW1.pdf`, `COMP0037_2627_CW2.pdf`) and the corresponding starter kits (`cw1_starter/`, `cw2_starter/`).

---

## 1. Assessment Structure & Module Alignment

COMP0037 is assessed via two **UCL AT10 / GN02 Group Courseworks** (100 marks each) unified under a single robotics narrative — **MedBot** (*Autonomous Hospital Pharmacy & Sterile Laboratory Assistant*) — using new environments and tasks distinct from the weekly labs while reusing the exact software stack from Weeks 01–09:

| Component | Weight | Weeks Covered | Total Marks | Scenario & Core Tasks | Python Env |
|:---|---:|:---|---:|:---|:---|
| **CW01** (`CW01/`) | **40%** | Weeks 01–05 | **100** (20 + 18 + 32 + 20 + 10) | Stationary/drifting hospital dispatch bandits (Q1), $8\times 12$ slippery ward MDP (Q2), 2-DoF sterile cabinet arm with circular centrifuge keep-out & Dual-Barrier CBF (Q3), `medbot_payload_trolley` CPU vs. GPU MuJoCo Warp & 3-param DR (Q4), sim-to-real deployment design (Q5) | `.venv_rl` (Q1–Q3), `.venv_sim` (Q4) |
| **CW02** (`CW02/`) | **60%** | Weeks 06–09 | **100** (25 + 25 + 25 + 25) | Dual-table mobile pick-and-place with `PandaOmron` (wheeled Omron base + 7-DoF Franka Panda arm): Behavioral Cloning & covariate shift (Q1), LLM task planning & typed mobile skill API (Q2), Track A VLA `NanoVLM` + Conditional DDPM Diffusion Chunk-MPC (Q3), Whole-Body CBF safety shield & 3-scenario red-teaming (Q4) | `.venv_robot` (Q1–Q4) |

---

## 2. How Starter Code and Reference Solutions Work

Both `CW01/` and `CW02/` follow the same architecture used in the weekly lab repository:
- **Student Starter Kits (`cw1_starter/`, `cw2_starter/`):** Self-contained environments and scaffolding scripts with explicit `TODO(CW1-...)` / `TODO(CW2-...)` stubs raising `NotImplementedError`. Every starter file can be executed directly for a quick environment self-test.
- **Instructor Reference Solutions (`cw1_solutions/`, `cw2_solutions/`):** Standalone scripts that dynamically monkey-patch the `TODO(...)` hooks in the starter modules at runtime **without modifying the starter files**, run all required coursework experiments, print summary tables, and save verified plots and rollout GIFs into `outputs/`.
- **LaTeX Specifications & Figure Generators:** Each coursework folder contains its `.tex` source, compiled `.pdf`, and a `generate_cw{1,2}_figures.py` script that regenerates all vector/rendered figures in `figures/`.

---

## 3. Directory Tree

```text
assessment/
├── README.md                              # This teaching-team guide
├── CW01/                                  # Coursework 1 (40%, Weeks 01–05, 100 marks)
│   ├── COMP0037_2627_CW1.tex              # LaTeX specification (UCL AT10/GN02 style)
│   ├── COMP0037_2627_CW1.pdf              # Compiled 9-page specification PDF (release to students)
│   ├── generate_cw1_figures.py            # Generates all 3 CW01 figures in figures/
│   ├── figures/                           # Vector PDF figures embedded in CW01 PDF
│   ├── cw1_starter/                       # Student starter kit (release to students)
│   └── cw1_solutions/                     # Instructor reference solutions + verified outputs/
└── CW02/                                  # Coursework 2 (60%, Weeks 06–09, 100 marks)
    ├── COMP0037_2627_CW2.tex              # LaTeX specification (UCL AT10/GN02 style)
    ├── COMP0037_2627_CW2.pdf              # Compiled 7-page specification PDF (release to students)
    ├── generate_cw2_figures.py            # Generates CW02 figures (incl. live MuJoCo render)
    ├── figures/                           # Vector/rendered PDF figures embedded in CW02 PDF
    ├── cw2_starter/                       # Student starter kit (release to students)
    └── cw2_solutions/                     # Instructor reference solutions + verified outputs/
```

---

## 4. Detailed Contents — Coursework 1 (`assessment/CW01/`)

### Specification & Figures
| File | Purpose |
|:---|:---|
| `COMP0037_2627_CW1.tex` / `.pdf` | Complete 9-page CW01 specification (Overview, Instructions, MedBot Use Case, Q1–Q5 = 100 marks, Getting Help, Submission & Feedback, Appendix A Group Contribution Table). |
| `generate_cw1_figures.py` | Regenerates `figures/fig1_hospital_ward.pdf` ($8\times 12$ ward map), `figures/fig2_slip_model.pdf` (stochastic wheel-slip dynamics), and `figures/fig3_sterile_arm_trolley.pdf` (2-DoF sterile arm + `medbot_payload_trolley` mechanical schematic). |

### Student Starter Kit (`CW01/cw1_starter/`)
| File | Contents & Student `TODO`s |
|:---|:---|
| `README.md` | Student-facing quick-start overview of the four CW01 starter modules. |
| `q1_bandits_hospital.py` | 8-bay hospital dispatch/charging bandit (`stationary` and `drifting` modes). `TODO(CW1-Q1b/c)`: `run_ucb1` and `run_epsilon_greedy_const_alpha`. |
| `q2_hospital_ward.py` | $8\times 12$ stochastic `HospitalWardEnv` (`start=(7,0)`, `goal=(0,11)`, sterile ICU `I`, automatic doors `D`, biohazard spill `B`). `TODO(CW1-Q2b)`: `train_tabular` (Q-Learning & SARSA). |
| `q3_sterile_arm.py` | 2-DoF planar `SterileArmEnv` (`discrete` 9-action & `continuous` velocity modes) with hard joint limits ($\pm 1.6\text{ rad}$) and circular centrifuge keep-out disk at $(0.45, 0.50)$, $r=0.16\text{ m}$. `TODO(CW1-Q3d)`: dual-barrier `cbf_safety_shield`. |
| `q4_payload_sim.py` | Custom MuJoCo MJCF model `medbot_payload_trolley` (wheeled chassis + hinged specimen mast). `TODO(CW1-Q4c)`: 3-parameter `sample_trolley_randomization` (`mast_damping` log/linear-uniform, `floor_friction`, `payload_mass`). |

### Instructor Solutions (`CW01/cw1_solutions/`)
| File | Description & Generated Outputs (`outputs/`) |
|:---|:---|
| `README.md` & `HOW_TO_RUN.md` | Complete local (Windows/Linux) and Google Colab execution guide with verified sanity-check numbers and troubleshooting. |
| `q1_bandits_solutions.py` | Patches Q1 TODOs; generates `q1_stationary.png` and `q1_drifting.png`. |
| `q2_hospital_ward_solutions.py` | Patches Q2 TODO; runs slip & $\varepsilon$ sweeps; generates `q2_slip_sweep.png` and `q2_eps_sweep.png`. |
| `q3_sterile_arm_solutions.py` | Compares Standard DQN vs. Double DQN vs. No-Target-Net, patches `cbf_safety_shield`, compares Unshielded vs. Soft-Penalty vs. Dual-Barrier CBF A2C (supports `--gif-seed`); generates `q3_dqn_comparison.png`, `q3_a2c_safety.png`, and `q3_sterile_arm_rollout.gif`. |
| `q4_payload_sim_solutions.py` | Benchmarks CPU MuJoCo vs. CUDA-Graph GPU MuJoCo Warp and runs 3-parameter DR; generates `q4_warp_throughput.png` and `q4_trolley_dr.png`. |

---

## 5. Detailed Contents — Coursework 2 (`assessment/CW02/`)

### Specification & Figures
| File | Purpose |
|:---|:---|
| `COMP0037_2627_CW2.tex` / `.pdf` | Complete 7-page CW02 specification (Overview, Instructions, `PandaOmron` Dual-Table Use Case, Q1–Q4 at 25 marks each = 100 marks, Getting Help, Submission & Feedback, Appendix A Group Contribution Table). |
| `generate_cw2_figures.py` | Regenerates `figures/fig1_mobile_arena.pdf` (live MuJoCo `agentview` render + top-down metric schematic of Table A/B) and `figures/fig3_wholebody_cbf.pdf` (coupled base + end-effector CBF safety boundaries). |

### Student Starter Kit (`CW02/cw2_starter/`)
| File | Contents & Student `TODO`s |
|:---|:---|
| `README.md` | Student-facing quick-start overview of the CW02 starter modules. |
| `mobile_pick_place_env.py` | Custom Robosuite 1.5.2 `MobilePickPlaceEnv` using built-in `PandaOmron` (`BASIC` 11-D whole-body controller: arm `OSC_POSE` + base `[vx, vy, wz]` + gripper) between side-by-side Table A ($y = -0.55\text{ m}$) and Table B ($y = +0.55\text{ m}$), plus `extract_proprio` (16-D). |
| `q1_mobile_lfd.py` | 8-phase scripted mobile-manipulation FSM expert (`MobilePickPlaceExpert`, 100% success) + `collect_demonstrations`. `TODO(CW2-Q1b/c)`: `train_bc` (with optional state-noise augmentation) and `evaluate_bc_policy`. |
| `q2_mobile_llm_planner.py` | Complete `MobileSkillLibrary` (`drive_to_station`, `stow_arm`, `pick_cube`, `place_cube`, `verify_grasp`). `TODO(CW2-Q2b/c)`: `validate_plan` (JSON schema + symbolic preconditions) and `MobileLLMPlanner.plan`. |
| `q3_mobile_vla.py` | Track A VLA scaffolding (`collect_vla_dataset` with $64\times 64$ RGB + 3 QA pairs/frame, `build_action_chunks`, `TinyViT`). `TODO(CW2-Q3b/c)`: `PixelShuffleProjector.forward`, `build_prefix_causal_mask`, `cosine_beta_schedule`, and `evaluate_chunk_mpc`. |
| `q4_mobile_cbf_integration.py` | 4-tier safe mobile-manipulation harness (`SafetyAuditEntry`, `WholeBodyCBFShield`). `TODO(CW2-Q4b/c)`: `WholeBodyCBFShield.filter_action` (base-to-table, aisle, EE height, velocity clamping, audit log) and `run_red_team_suite`. |

### Instructor Solutions (`CW02/cw2_solutions/`)
| File | Description & Generated Outputs (`outputs/`) |
|:---|:---|
| `README.md` & `HOW_TO_RUN.md` | Execution commands, verified reference numbers, and PDF regeneration instructions. |
| `q1_mobile_lfd_solutions.py` | Patches Q1 TODOs; evaluates BC dataset-size scaling, Standard BC vs. Noise-Augmented BC, and OOD cube perturbations; generates `q1_lfd_comparison.png` and `q1_mobile_pick_place.gif`. |
| `q2_mobile_llm_solutions.py` | Patches Q2 TODOs; runs 5-case precondition validator suite and closed-loop dual-table transfer; generates `q2_llm_plan_trace.png` and `q2_llm_rollout.gif`. |
| `q3_mobile_vla_solutions.py` | Patches Q3 TODOs; trains `NanoVLM` from scratch and Conditional DDPM Diffusion Head (30 vs. 160 epochs, Proprio-Only vs. Proprio + VLM, $k=1$ vs. $k=4$); generates `q3_vla_training.png`, `q3_chunk_mpc.png`, and `q3_vla_rollout.gif`. |
| `q4_mobile_cbf_solutions.py` | Patches Q4 TODOs; compares Unshielded vs. Whole-Body CBF across Nominal, Adversarial Ram+Slam, and Biased/Noisy Perception scenarios; generates `q4_cbf_redteam.png`, `q4_cbf_nominal.gif`, `q4_cbf_adversarial.gif`, and `q4_cbf_noisy.gif`. |

---

## 6. Moodle Release & Recompilation Checklist

1. **Release to students on Moodle:**
   - For **CW01:** upload `CW01/COMP0037_2627_CW1.pdf` and a zip of `CW01/cw1_starter/` only.
   - For **CW02:** upload `CW02/COMP0037_2627_CW2.pdf` and a zip of `CW02/cw2_starter/` only.
2. **Recompiling a specification after `.tex` edits (run twice for cross-references):**
   ```powershell
   pdflatex --disable-installer -interaction=nonstopmode -output-directory=assessment/CW01 assessment/CW01/COMP0037_2627_CW1.tex
   pdflatex --disable-installer -interaction=nonstopmode -output-directory=assessment/CW01 assessment/CW01/COMP0037_2627_CW1.tex
   ```
   *(Replace `CW01` / `CW1` with `CW02` / `CW2` for Coursework 2.)*

