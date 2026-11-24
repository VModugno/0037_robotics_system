# Branch Overview & Differences vs. `main` (`COMP0037` Labs, Weeks 01–09)

> **Audience:** COMP0037 Teaching Team  
> **Purpose:** This document summarizes all modifications made in this branch relative to the original `main` branch of the `0037_robotics_system` labs repository, so fellow lecturers and teaching assistants can review every change at a glance.

---

## 1. High-Level Summary of Changes

All modifications fall into four clean categories while **strictly preserving every student `TODO` signature, `pixi` environment manifest, and default script output path**:

1. **Assessment Policy & Deliverables Alignment (17 files):**
   - Removed obsolete references to a graded 6-page PDF lab report across `README.md`, all nine `LAB.md` files, and starter script docstrings/prints.
   - Standardized the end-of-lab sections on **Deliverables & Discussion Questions** (`lab notes` + checklist + reproducibility block).
   - Removed the leaked `## Instructor solutions` section from `README.md` so the student-facing README does not reference instructor-only instructions.
2. **Bug Fixes & Runtime / Colab Compatibility (6 starter files + 1 solution file):**
   - Fixed Linux/Colab `libmujoco.so` plugin resolution in Week 05 (`sim_models.py`).
   - Fixed the Week 05 Domain Randomization plot to start at the true $t = 0$ initial state (`+0.300 rad`, `std = 0.0000`) and added a per-world `(damping, friction)` legend when `nworld <= 8` (`domain_randomization.py`).
   - Added a defensive `int(...)` joint-type compatibility shim in Weeks 06–09 to prevent enum-comparison `AssertionError` if MuJoCo builds get mixed on Colab.
   - Added explicit `np.clip(chunk[0], -1.0, 1.0)` actuator clamping in `solutions/week08_vla_solutions.py` (`evaluate_policy`) to avoid unbounded DDPM samples during closed-loop evaluation.
   - Added a clear Colab warning in `README.md` regarding the MuJoCo version split between **Week 05** (`mujoco-warp`, MuJoCo $\ge 3.5$) and **Weeks 06–09** (`robosuite==1.5.2`, MuJoCo $3.4.0$).
3. **Pedagogical Addition (1 file):**
   - Added `week06_lfd/ROBOT_CONTROL_PRIMER.md` explaining 7-DoF Operational Space Control (`OSC_POSE`), the 7-D action vector, and the 5-phase grasping FSM for students transitioning from 2-DoF arms to the Franka Panda.
4. **Explicitly Preserved / Unchanged vs. `main`:**
   - **Zero student `TODO` blocks modified or overwritten** across Weeks 01–09.
   - **Environment manifests unchanged:** `pixi.toml`, `pixi.lock`, `.gitignore`, and all nine `requirements.txt` files are identical to `main`.
   - **Default working-directory paths unchanged:** no `--out-dir` routing changes or extra animation/visualizer scripts were added to the student folders.
   - **`week01_bandits/mdp_formulation.py` goal cell preserved at `G = (5, 8)`** so student self-checks match `week01_bandits/LAB.md`.

---

## 2. File-by-File Changelog vs. `main`

### Root Documentation
| File | Change vs. `main` |
|:---|:---|
| `README.md` | Replaced graded 6-page report wording with **Deliverables & Self-Assessment**; removed the leaked `## Instructor solutions` section; added Colab runtime note on the Week 05 (MuJoCo $\ge 3.5$) vs. Weeks 06–09 (MuJoCo $3.4$) split. |

### Weekly Lab Guides (`LAB.md`)
| Files | Change vs. `main` |
|:---|:---|
| `week01_bandits/LAB.md` through `week09_integration/LAB.md` (all 9 files) | Renamed Section 6 to **Deliverables & Discussion Questions**; removed `(<= 6 pages, PDF)` report requirements; replaced `"in your report"` / `"Report rules"` with `"in your lab notes"` / `"Scientific rigor"`. All experiments, TODO tables, and self-check reference numbers remain intact. |

### Starter Code — Minor Wording Updates Only (Zero Logic Changes)
| File | Change vs. `main` |
|:---|:---|
| `week01_bandits/bandits.py` | Docstring/print wording: `"in your report"` $\to$ `"in your lab notes"`. |
| `week02_tabular_rl/run_lab.py` | Print wording: `"Report Q3"` $\to$ `"Lab discussion question"`. |
| `week03_dqn/dqn.py` | Comment/print wording: `"report Q3/Q4"` $\to$ `"discussion Q3/Q4"`. |
| `week04_actor_critic/actor_critic.py` | Comment/print wording: `"report Q4"` $\to$ `"discussion Q4"`. |
| `week05_simulation/domain_randomization.py` | Wording: `"report both numbers"` $\to$ `"record both numbers"` (plus plot fixes below). |
| `week08_vla/evaluate_vla.py` | Comment wording: `"say so in your report"` $\to$ `"note this in your findings"`. |
| `week09_integration/pipeline.py` | Comment wording: `"report needs the table format"` $\to$ `"analysis requires this table format"`. |

### Starter Code — Bug Fixes & Compatibility Shims
| File | Change vs. `main` |
|:---|:---|
| `week05_simulation/sim_models.py` | Preloads `libmujoco.so*` with `ctypes.RTLD_GLOBAL` before `import mujoco` on Linux/Colab so bundled MuJoCo plugins load without `OSError`. |
| `week05_simulation/domain_randomization.py` | Captures and prepends the true $t = 0$ initial pole angle (`+0.300 rad`, `std = 0.0000`) to `traj` before stepping; adds per-world `(damping, friction)` legend to `plot_trajectories()` when `nworld <= 8`. |
| `week06_lfd/collect_demos.py` | Defensive `int(...)` joint-type comparison shim in Robosuite's `MjModel.get_joint_qpos_addr` / `get_joint_qvel_addr` to avoid enum mismatch crashes if MuJoCo builds are mixed on Colab. |
| `week07_llm_planning/robot_api.py` | Same defensive Robosuite `int(...)` joint-type compatibility shim. |
| `week08_vla/vlm_data.py` | Same defensive Robosuite `int(...)` joint-type compatibility shim. |
| `week09_integration/pipeline.py` | Same defensive Robosuite `int(...)` joint-type compatibility shim. |

### New Pedagogical & Student Setup Files
| File | Description |
|:---|:---|
| `COLAB_GUIDE.md` | Student-facing Google Colab setup & execution guide covering Weeks 01–09 (CPU/T4 GPU, split Cell 3A vs. Cell 3B for Week 05 MuJoCo Warp vs. Weeks 06–09 Robosuite 1.5.2, zero instructor `solutions/` references). |
| `week06_lfd/ROBOT_CONTROL_PRIMER.md` | Short reference primer introducing 7-DoF Franka Panda `OSC_POSE` control, the 7-D action vector, and the 5-phase scripted grasping FSM. |

### Reference Solutions (`solutions/`)
| File | Change vs. `main` |
|:---|:---|
| `solutions/week08_vla_solutions.py` | Added `np.clip(chunk[0], -1.0, 1.0)` in `_evaluate_policy` before stepping the Robosuite environment so DDPM-sampled actions always stay within actuator limits. |

---

## 3. Verification Status

- **Syntax check:** All modified Python files pass `py_compile` cleanly.
- **Pedagogical integrity:** All student `TODO` blocks and `NotImplementedError` stubs remain untouched; no instructor-only course outlines or solutions instructions leak into `README.md` or `LAB.md`.

