# COMP0037 2026/27 — Coursework 1 Reference Solutions (`cw1_solutions/`)

**Instructor-only — do not ship to students.**

Following the same architecture as `solutions/weekNN_*_solutions.py`, these scripts dynamically patch the `TODO(CW1-...)` hooks in `assessment/cw1_starter/` without modifying the starter files, run all required Coursework 1 experiments, print reference tables, and save diagnostic figures under `assessment/cw1_solutions/outputs/`.

## Running the Reference Solutions

From the repository root:

```powershell
# Q1 (Bandits: stationary horizon sweep + drifting non-stationary comparison)
.venv_rl\Scripts\python assessment\cw1_solutions\q1_bandits_solutions.py

# Q2 (HospitalWardEnv: Q-Learning vs. SARSA across slip_prob and epsilon sweeps)
.venv_rl\Scripts\python -X utf8 assessment\cw1_solutions\q2_hospital_ward_solutions.py

# Q3 (SterileArmEnv: DQN vs. Double DQN + Dual-Barrier CBF-Shielded A2C)
# Add --fast for a quick 1-seed smoke test, or --gif-seed <N> for a specific rollout episode
.venv_rl\Scripts\python assessment\cw1_solutions\q3_sterile_arm_solutions.py

# Q4 (MedBot Payload Trolley: CPU vs. GPU Warp benchmark + 3-param DR)
.venv_sim\Scripts\python assessment\cw1_solutions\q4_payload_sim_solutions.py
```

## Generated Reference Artifacts (`outputs/`)
- `q1_stationary.png`, `q1_drifting.png`
- `q2_slip_sweep.png`, `q2_eps_sweep.png`
- `q3_dqn_comparison.png`, `q3_a2c_safety.png`, `q3_sterile_arm_rollout.gif`
- `q4_warp_throughput.png`, `q4_trolley_dr.png`

