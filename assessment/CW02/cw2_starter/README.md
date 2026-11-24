# COMP0037 2026/27 — Coursework 2 Starter Code (`cw2_starter/`)

Starter environments and scaffolding for the **MedBot Mobile Manipulator (`PandaOmron`) Dual-Table Sample Transfer** coursework (Weeks 06–09):

- `mobile_pick_place_env.py` — Custom Robosuite 1.5.2 `MobilePickPlaceEnv` with `PandaOmron` (wheeled Omron base + 7-DoF Franka Panda arm) operating between side-by-side Table A (pickup, $y = -0.55\text{ m}$) and Table B (drop-off, $y = +0.55\text{ m}$).
- `q1_mobile_lfd.py` — 8-phase scripted mobile-manipulation FSM expert + `TODO(CW2-Q1b/c)` hooks for Behavioral Cloning (BC) and closed-loop evaluation.
- `q2_mobile_llm_planner.py` — Typed Mobile-Manipulation Skill API (`drive_to_station`, `stow_arm`, `pick_cube`, `place_cube`, `verify_grasp`) + `TODO(CW2-Q2b/c)` hooks for schema/precondition validation and LLM task planning with recovery.
- `q3_mobile_vla.py` — Track A VLA scaffolding (`collect_vla_dataset`, `TinyViT`, `PixelShuffleProjector`, `ConditionalDiffusionHead`) + `TODO(CW2-Q3b/c)` hooks for nanoVLM and receding-horizon Diffusion Chunk-MPC.
- `q4_mobile_cbf_integration.py` — 4-tier safe mobile-manipulation integration + `TODO(CW2-Q4b/c)` hooks for `WholeBodyCBFShield` (coupled base + EE barriers) and 3-scenario red-teaming.

Run each file directly in `.venv_robot` to verify the environment and self-test before implementing your coursework extensions.

