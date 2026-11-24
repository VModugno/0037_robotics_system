# COMP0037 — Student Google Colab Setup & Execution Guide (Weeks 01–09)

This guide gives you ready-to-copy cells to run **all 9 weekly labs** of COMP0037 on **Google Colab** (Free Tier with CPU or T4 GPU) without installing anything locally.

> **Tip:** Always edit and run your own weekly starter scripts (`week01_bandits/` through `week09_integration/`). Follow each week's `LAB.md` for the exact `TODO` blocks and discussion questions.

---

## 1. Initial Colab Setup

Open a notebook on [Google Colab](https://colab.research.google.com/) and select **Runtime → Change runtime type**:
- **CPU** is enough for Weeks 01, 02, 06, 07, and 09 (and works for Weeks 03, 04, and 08 Track A).
- **T4 GPU** is required for **Week 05 (MuJoCo Warp)** and recommended for faster training in **Weeks 03, 04, and 08**.

### Cell 1 — Load Your Lab Folder
Choose **Option A** (Google Drive, recommended so your code edits are saved) or **Option B** (direct ZIP upload):

```python
# Option A: Google Drive (recommended)
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/0037_robotics_system
```

```bash
# Option B: Direct ZIP upload to /content
!unzip -q /content/0037_robotics_system.zip -d /content/
%cd /content/0037_robotics_system
```

### Cell 2 — Headless Linux Graphics Libraries (Needed for MuJoCo / Robosuite)
```bash
!apt-get update -qq
!apt-get install -y -qq libosmesa6-dev libgl1-mesa-glx libglfw3 libegl1-mesa-dev patchelf xvfb
```

---

## 2. Python Packages — Choose Your Runtime Track

> **Important (MuJoCo Version Split):**
> - **Week 05** (`mujoco-warp`) requires **MuJoCo $\ge 3.5$**.
> - **Weeks 06–09** (`robosuite==1.5.2`) requires **MuJoCo $3.4.0$** (`MjData.qM` was removed in 3.5).
>
> Do **not** install both in the same runtime. Use **Cell 3A** for Weeks 01–05, and **Cell 3B** in a fresh runtime (or after restarting) for Weeks 06–09.

### Cell 3A — Install Packages for Weeks 01–05
```bash
!pip install -q "numpy>=1.26,<2.0" "matplotlib>=3.8" "torch>=2.2" "h5py>=3.10" pillow tqdm
# Optional Week 01 GPU bandits (CuPy) & required Week 05 GPU simulator (Warp + MuJoCo Warp)
!pip install -q cupy-cuda12x "warp-lang==1.18.0" mujoco-warp
```
*(After this cell finishes, click **Runtime → Restart session**, then run Cell 4.)*

### Cell 3B — Install Packages for Weeks 06–09 (Robosuite)
```bash
!pip uninstall -y -q mujoco mujoco-warp warp-lang
!pip install -q "numpy==1.26.4" "mujoco==3.4.0" "robosuite==1.5.2" "torch>=2.2" "matplotlib>=3.8" "h5py>=3.10" pillow tqdm
```
*(Ignore any red Colab pip warnings about unrelated pre-installed packages like `jax` or `geopandas`, then click **Runtime → Restart session** and run Cell 4.)*

---

## 3. Cell 4 — Environment Verification & Inline Plot Helper

Run this cell after restarting the session:

```python
import os, glob, site, ctypes
os.environ.setdefault("MUJOCO_GL", "egl")

# Preload libmujoco.so with RTLD_GLOBAL on Linux/Colab
for p in site.getsitepackages():
    for so in glob.glob(os.path.join(p, "mujoco", "libmujoco.so*")):
        ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        os.environ["LD_LIBRARY_PATH"] = f"{os.path.dirname(so)}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import numpy as np
import torch
print("NumPy:", np.__version__, "| PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())

from IPython.display import Image, display
def show_media(path: str) -> None:
    if os.path.exists(path):
        display(Image(filename=path))
    else:
        print(f"File not found: {path}")
```

---

## 4. Week-by-Week Starter Commands

Run each week from inside its folder (or adjust paths accordingly) after completing the `TODO` blocks in that week's `LAB.md`:

### Week 01 — Bandits & MDPs (`week01_bandits/`)
```bash
%cd week01_bandits
!python mdp_formulation.py
!python bandits.py
```
```python
show_media("regret_comparison.png")
show_media("optimal_action.png")
```

### Week 02 — Tabular RL: Q-Learning vs. SARSA (`week02_tabular_rl/`)
```bash
%cd ../week02_tabular_rl
!python run_lab.py --env both
```
```python
show_media("warehouse_learning.png")
show_media("cliff_learning.png")
```

### Week 03 — Deep Q-Networks (`week03_dqn/`)
```bash
%cd ../week03_dqn
!python dqn.py --episodes 300 --device cuda
```
```python
show_media("dqn_learning.png")
```

### Week 04 — Continuous Actor-Critic & CBF Safety Shield (`week04_actor_critic/`)
```bash
%cd ../week04_actor_critic
!python actor_critic.py --episodes 400 --device cuda
```
```python
show_media("a2c_comparison.png")
```

### Week 05 — Simulation at Scale: MuJoCo CPU vs. Warp GPU & DR (`week05_simulation/`)
*(Requires **T4 GPU** + **Cell 3A**. Verify `wp.get_device()` reports `cuda:0`.)*
```bash
%cd ../week05_simulation
!python benchmark_mujoco_cpu.py
!python benchmark_mujoco_warp.py --nworlds 1,4,16,64,256
!python domain_randomization.py --nworld 32 --steps 500
```
```python
show_media("warp_throughput.png")
show_media("domain_randomization.png")
```

### Week 06 — Learning from Demonstration (`week06_lfd/`)
*(Switch to **Cell 3B** + **Restart session** before running Weeks 06–09.)*
```bash
%cd ../week06_lfd
!python collect_demos.py --episodes 20 --no-cameras
!python extract_obs.py --no-cameras
!python bc.py --epochs 100 --eval-episodes 10
```

### Week 07 — LLM Task Planning on a Typed Skill API (`week07_llm_planning/`)
```bash
%cd ../week07_llm_planning
!python robot_api.py --demo
!python llm_planner.py --provider mock --dry-run
!python llm_planner.py --provider mock --task "pick up the cube"
```

### Week 08 — Vision-Language-Action Models: Track A (`week08_vla/`)
```bash
%cd ../week08_vla
!MUJOCO_GL=egl python vlm_data.py --episodes 10 --directory vlm_data
!python nanovlm_from_scratch.py --data vlm_data --epochs 15
!python diffusion_action_head.py --data vlm_data --epochs 150
!python diffusion_action_head.py --data vlm_data --epochs 150 --vlm-checkpoint nanovlm.pt
```

### Week 09 — Safe System Integration (`week09_integration/`)
```bash
%cd ../week09_integration
!python safety_filter.py
!python pipeline.py --task "pick up the cube" --episodes 5
!python pipeline.py --task "slam the cube" --episodes 2
!python pipeline.py --task "slam the cube" --episodes 2 --unsafe
```

---

## 5. Common Colab Troubleshooting

| Symptom | Cause | Fix |
|:---|:---|:---|
| `AttributeError: 'MjData' object has no attribute 'qM'` in Weeks 06–09 | MuJoCo $\ge 3.5$ left over from Week 05 | Run **Cell 3B** (`mujoco==3.4.0`, `robosuite==1.5.2`) and **Restart session**. |
| `wp.get_device()` reports `cpu` in Week 05 | Colab T4 (CUDA 12.4) vs. CUDA 13 `warp-lang` wheel | Pin `warp-lang==1.18.0` (as in Cell 3A) and **Restart session**. |
| `OSError: libmujoco.so` cannot open shared object file | Linux dynamic loader path not preloaded | Run **Cell 4** before importing `mujoco` or `robosuite`. |
| `GLEW` / `GLFW` error when rendering images in Week 08 | Missing headless EGL backend | Ensure `os.environ["MUJOCO_GL"] = "egl"` is set **before** importing `robosuite`. |

