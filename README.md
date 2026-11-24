# COMP0037 Labs — Reinforcement Learning & Robot Learning (Weeks 1–9)

Nine lab sessions, one folder per week, matching the module's tentative
schedule. Every week has the same shape: a `LAB.md` (what to do, experiments,
self-check with reference numbers, and a **deliverables & discussion section**)
plus one or more `.py` files with `TODO` blocks you implement. Simpler labs run on CPU with numpy; the RL/VLA labs use
PyTorch; weeks 5–9 use MuJoCo / MuJoCo Warp / robosuite / VLA Foundry.

## Index

| Week | Topic | Folder | pixi env | Hardware | Key scripts |
|------|-------|--------|----------|----------|-------------|
| 1 | MDPs, bandits, exploration | `week01_bandits/` | `week01` | CPU | `mdp_formulation.py`, `bandits.py` |
| 2 | Tabular model-free RL | `week02_tabular_rl/` | `week02` | CPU | `gridworld.py`, `agents.py`, `run_lab.py` |
| 3 | DQN (replay + target nets) | `week03_dqn/` | `week03` | CPU (GPU ok) | `reach_env.py`, `dqn.py` |
| 4 | Actor-Critic + safe RL (CBF) | `week04_actor_critic/` | `week04` | CPU (GPU ok) | `arm2d_env.py`, `actor_critic.py` |
| 5 | Simulation & sim-to-real | `week05_simulation/` | `week05` | CPU + GPU (Warp) | `benchmark_mujoco_cpu.py`, `benchmark_mujoco_warp.py`, `domain_randomization.py` |
| 6 | Learning from demonstration | `week06_lfd/` | `week06` | CPU | `collect_demos.py`, `extract_obs.py`, `bc.py`, `airl.py` |
| 7 | LLMs in robotics (task planning) | `week07_llm_planning/` | `week07` | CPU | `robot_api.py`, `llm_planner.py` |
| 8 | VLAs: nanoVLM → diffusion head; Foundry | `week08_vla/` | `week08` | CPU (GPU for Track B) | `vlm_data.py`, `nanovlm_from_scratch.py`, `diffusion_action_head.py`, `evaluate_vla.py`, `finetune_vla.py` |
| 9 | System integration (mini-project) | `week09_integration/` | `week09` | CPU | `safety_filter.py`, `pipeline.py` |

Week 10 has no lab (project/assessment period).

## Setup with pixi

[pixi](https://pixi.sh) (≥ 0.80) manages one isolated environment per week
from the single `pixi.toml` manifest in this folder. From `labs/`:

```bash
pixi install -e weekNN          # create/update the environment
pixi run -e weekNN python <script> [flags]
```

From inside a week folder, use the manifest path:

```bash
pixi run --manifest-path ../pixi.toml -e weekNN python <script>
```

The manifest also sets two Windows-friendly activation variables:
`KMP_DUPLICATE_LIB_OK=TRUE` (conda MKL vs torch OpenMP clash) and
`PYTHONUTF8=1` (Windows console Unicode). If you don't use pixi, each week
folder has a `requirements.txt` for pip.

## Running on Colab

Every `LAB.md` has a ready Colab section. The short version:

1. **Weeks 1–4, 6, 7, 9** — CPU runtime is enough. Upload the week folder,
   `!pip install -r requirements.txt`, run the scripts from the LAB.md cells.
2. **Week 3/4 GPU speedups** — switch to a T4 runtime and pass `--device cuda`.
3. **Week 1 GPU bandits** — `--gpu` uses CuPy on a T4 (optional).
4. **Week 5 (MuJoCo Warp)** — needs CUDA. Two gotchas:
   `os.environ.setdefault("MUJOCO_GL", "egl")` for rendering-free ops, and the
   `warp-lang` wheel from PyPI is built for CUDA 13.x (driver R580+); on a
   Colab T4 (driver 550 / CUDA 12.4) install the `+cu12` wheel from the
   [NVIDIA warp releases](https://github.com/NVIDIA/warp/releases) (or pin
   `warp-lang==1.18.0`), then check `wp.get_device()` says `cuda:0`.
5. **Week 6–9 (robosuite)** — set `MUJOCO_GL=egl` *before* importing
   robosuite when collecting images (week 8); everything else runs headless
   without it. **Note:** Week 5 (`mujoco-warp`, MuJoCo ≥ 3.5) and Weeks 6–9
   (`robosuite==1.5.2`, MuJoCo 3.4) require incompatible MuJoCo versions — use
   a fresh Colab runtime (or uninstall `mujoco`/`mujoco-warp`, remove the
   `mujoco` package folder, reinstall `numpy==1.26.4 mujoco==3.4.0 robosuite==1.5.2`,
   and restart the session) when moving from Week 5 to Weeks 6–9.
6. **Week 8 Track B (VLA Foundry)** — Colab/server with a ≥16 GB GPU only:
   Foundry installs via git + `uv` (Python 3.12), is not pip-installable, and
   needs `amp` (not `amp_bf16`) precision on T4s. The LAB.md spells out the
   recipe; Tracks A (local) and B (GPU) are separate.

## Deliverables & Self-Assessment (every week, same rules)

Each `LAB.md` ends with a deliverables and discussion section: checklist of
artifacts, plots to generate, week-specific analysis questions, and a
reproducibility block (machine, env, exact commands, seeds). Reference
numbers quoted in the "Self-check" sections are from seeded runs on the
instructor machine — treat them as sanity anchors, not targets to reproduce
digit-for-digit (several labs note where RNG conventions can shift decimals).

## Toolchain notes & troubleshooting

- **Windows + robosuite (weeks 6–9):** create `C:\tmp` before importing
  robosuite; if MuJoCo's DLL fails to load, copy `mujoco.dll` from the mujoco
  package into `site-packages\robosuite\utils`. The lab code handles `C:\tmp`
  for you.
- **Version pins that matter:** `robosuite==1.5.2` needs `mujoco>=3.3,<3.5`
  (3.5 removed the `qM` attribute its OSC controller reads) and `numpy<2`
  (its `mink` dependency). `mujoco-warp>=4.0` needs `mujoco>=3.13` and
  `warp-lang>=1.15`.
- **Week 5 on locked-down Windows:** if `warp` fails with a policy error
  (`WinError 4551` — Application Control blocking `warp-clang.dll`), that is a
  machine restriction, not a bug: run Part B/C on Colab (the script prints
  this hint itself).
- **Week 8 footgun:** `nn.MultiheadAttention` bool masks mean different
  things in torch's train vs eval paths (a mask that trains fine can NaN at
  eval). The lab teaches float `0/-inf` masks — see its LAB.md E5.
- **torch ≥ 2.6:** `torch.load` defaults to `weights_only=True`; labs that
  load their own checkpoints with numpy arrays pass `weights_only=False`
  explicitly (flagged in code).
