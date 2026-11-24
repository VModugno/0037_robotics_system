# Week 06 Lab — Learning from Demonstration (LfD)

**Task:** collect demonstrations from a scripted expert in the robosuite `Lift`
task, learn a policy from them, and study what breaks.

You will build the full demonstration pipeline the way practitioners actually
run it:

```
scripted expert -> raw (state, action) demos -> replayed observations
                -> behaviour cloning (BC) -> (advanced) adversarial IRL
```

**Key learning outcomes.** After this lab you can:

- drive a robot with a *scripted expert* (a hand-written waypoint controller —
  no teleoperation, no RL);
- explain why robosuite records **states, not observations**, and how to
  reconstruct observations from states (`sim.set_state_from_flattened` +
  `sim.forward()` + `_get_observations(force_update=True)` — the same trick
  robomimic's `dataset_states_to_obs.py` uses);
- train and evaluate a behaviour-cloned policy **closed-loop**, and diagnose
  covariate shift from its failure cases;
- (advanced) explain the adversarial IRL (AIRL/GAIL) idea and why it is
  slower and less stable than BC on small datasets.

---

## 1. Setup

From this folder (`week06_lfd/`):

```bash
pixi run --manifest-path ../pixi.toml -e week06 python collect_demos.py --help
```

The `week06` pixi environment pins `robosuite==1.5.2`, `mujoco>=3.3,<3.5`
(robosuite 1.5.2 predates the MuJoCo `qM`→`M` rename; mujoco ≥ 3.5 crashes
its OSC controller), and `numpy<2` (its `mink` dependency requires it).

**Windows note:** the scripts create `C:\tmp` automatically — MuJoCo's Windows
GL binding needs it. If you see `ImportError`/`OSError` mentioning
`mujoco.dll`, the pixi environment is the supported fix (pypi-layout mujoco).

**Colab (GPU or CPU runtime):** set the GL backend *before* robosuite is
imported — robosuite's `binding_utils` auto-selects `egl` on Colab, which is
what we want:

```python
import os
os.environ["MUJOCO_GL"] = "egl"   # run this cell BEFORE the pip install cell re-imports
```

```bash
!pip install robosuite==1.5.2 "mujoco>=3.3,<3.5" "numpy<2" h5py torch matplotlib
```

Upload the `week06_lfd/` folder (or clone the repo) and run the commands
below. A CPU runtime works for this lab — rendering is off by default.

**Hardware:** CPU-only is enough (~1 GB RAM). GPU changes nothing here.

---

## 2. The task and the data format

`Lift`: a Franka Panda must lift a cube ~4 cm off the table. Success
(`env._check_success()`) latches the moment the cube is high enough.

Robosuite's `DataCollectionWrapper` records, per episode: the MJCF model
(`model.xml`), the flattened simulator state *at reset and after every
action*, every action, and whether the episode was successful. It stores
states — **not** observations. States are compact and complete; observations
are a *function of state* that you can recompute at any time (T3). This is
the same recording format used by robomimic and most of the imitation
ecosystem.

`gather_demonstrations_as_hdf5` merges per-episode folders into one HDF5 with
the **raw robomimic layout**:

```
data/
  demo_1/            <- NOTE: 1-indexed in the raw format
    states  (T+1, D) flattened MjSimState   (gather trims the last one,
              so states[t] is the state BEFORE actions[t])
    actions (T, 7)
```

**Gotcha:** only episodes the wrapper saw succeed are merged. A broken
expert produces an *empty* dataset, silently. Watch the console.

---

## 3. Your TODOs

| # | File | Function | What you build |
|---|------|----------|----------------|
| T1 | `collect_demos.py` | `expert_action` | P-controller action toward a waypoint |
| T2 | `collect_demos.py` | `ExpertController.next_action` | waypoint phase machine (hover → descend → grasp → lift → hold), tracking the cube **live** |
| T3 | `extract_obs.py`   | `replay_observation` | state → observation replay |
| T4 | `bc.py`            | `build_dataset` | HDF5 → (X, A) arrays |
| T5 | `bc.py`            | `BCNet.forward` | policy MLP forward |
| T6 | `bc.py`            | `train_step` | one BC optimisation step |
| T7 | `bc.py`            | `evaluate_policy` | closed-loop success-rate evaluation |
| T8 | `airl.py`          | `discriminator_loss` | BCE for expert/novice classification |
| T9 | `airl.py`          | `adversarial_reward` | r = −log(1 − D + ε) |
| T10| `airl.py`          | `reinforce_update` | REINFORCE with discounted reward-to-go |

T8–T10 are the advanced part — do them only after BC works.

---

## 4. Experiments

Run each step in order (all from `week06_lfd/`):

```bash
# 1. collect 20 expert demos (camera-less; ~1 min)
pixi run --manifest-path ../pixi.toml -e week06 python collect_demos.py \
    --episodes 20 --no-cameras

# 2. replay states to observations (~5 s)
pixi run --manifest-path ../pixi.toml -e week06 python extract_obs.py \
    --no-cameras

# 3. behaviour-clone (~20 s CPU + evaluation)
pixi run --manifest-path ../pixi.toml -e week06 python bc.py \
    --epochs 100 --eval-episodes 10

# 4. (advanced) adversarial IRL, seeded from the BC policy
pixi run --manifest-path ../pixi.toml -e week06 python airl.py \
    --iterations 20
```

Then run the **analysis experiments**:

- **E1 — expert quality.** How does the expert success rate change with
  `GRIP_STEPS` (40 → 20, 80)? With `TOL` (0.02 → 0.01, 0.04)? Record rates.
- **E2 — dataset size.** Re-train BC with 5 / 10 / 20 demos
  (`--episodes` at collection time; keep the BC flags fixed). Plot
  closed-loop success vs number of demos.
- **E3 — covariate shift.** The BC policy succeeds on most episodes but
  fails some. Print the observation features of a *failing* episode's first
  10 steps vs the expert's. Where does the policy leave the expert
  distribution? (You can log `obs_features(obs)` inside `evaluate_policy`.)
- **E4 — camera demos (optional).** Re-run collect + extract *without*
  `--no-cameras` (images at 128 px, slower). Confirm the `*_image` keys
  appear in the obs HDF5. BC in this lab ignores them — what would you have
  to change to clone from images?

---

## 5. Self-check

Reference numbers from the instructor machine (20 episodes, camera-less,
seed 0, `pixi run -e week06`):

```
expert success rate: 20/20      mean episode length: 61.8 steps (wall ~52 s)
episodes: 20   total steps: 1236      (extract, wall ~5 s)
dataset: 1236 steps, obs dim 12, action dim 7
early stop at epoch 78 (best val 0.008283)     training wall 16.5 s
policy success rate: 8/10  mean steps: 83.0
```

Your exact numbers will differ (RNG conventions, CPU speed) but should be
in the same regime: **expert ≥ 90 %**, BC **6–10 / 10** with successful
episodes typically *faster* than the expert's (a cloned P-controller is more
aggressive than the waypoint scheduler that produced the data — a nice
discussion point). If the expert is below ~50 %, debug T1/T2 first: the most
common cause is *not tracking the cube live* (the stale-waypoint expert
lifts empty-handed and scores 9/20 on this machine).

AIRL (3-iteration smoke run, seed 0):

```
iter  D_acc   nov_r  BC-eval
   1 53.28%   0.042 0/5      (g_loss +0.9282)
   2 72.19%   0.067 0/5      (g_loss +1.3072)
   3 76.95%   0.067 0/5      (g_loss +1.0364)
```

The 0/5 column is with the exploration noise AIRL adds on top of the policy
(std 0.3 on a tanh head is a lot); with 20 iterations the generator starts
moving `nov_r` up. Expect AIRL to be *worse* than BC here — that is the
finding (see Q6).

---

## 6. Deliverables & Discussion Questions

Record your experimental findings and answer the following questions in your lab notes:

**Deliverables checklist**

- [ ] `demos/lift_demos.hdf5` (raw) and `demos/lift_demos_obs.hdf5` sizes +
      episode counts
- [ ] `bc_training.png` and your BC eval console output
- [ ] E1–E3 outputs (tables or plots; axes labelled)
- [ ] (advanced) AIRL iteration table

**Questions**

- **Q1.** The wrapper stores states, not observations. Give two reasons this
  is the better engineering choice, and one situation where it fails
  (hint: think camera parameters and sensor noise).
- **Q2.** Why is `sim.forward()` needed between `set_state_from_flattened`
  and `_get_observations`? What would you observe if you forgot it?
- **Q3.** Why does `gather_demonstrations_as_hdf5` drop *unsuccessful*
  episodes? What bias does that introduce to the dataset (and to what BC
  learns)?
- **Q4.** Your BC policy succeeds in fewer steps than the expert (26–33 vs
  ~62 in the reference run). Explain why, referencing the waypoint
  scheduler vs a reactive policy.
- **Q5.** Describe one failing BC episode: at which step does the policy
  leave the expert data distribution, and what did the arm do? This is
  *covariate shift* — define the term in your own words.
- **Q6.** AIRL was slower and no better than BC here. Explain the intended
  advantage of IRL over BC (what does a *reward* give you that supervised
  action targets do not?) and why 20 demos is too few for it to show.
- **Q7.** The expert tracks the cube live (step 0 in T2). Why does the
  episode-start-only variant fail ~half the time? What general lesson does
  this teach about hand-engineered policies?

**Reproducibility:** state the seeds (`--seed 0` everywhere), the exact
commands, robosuite/mujoco versions from the pixi env, and your machine
(CPU/GPU, OS). Note any deviations.

---

## 7. Going further

- Collect demos **with cameras** and clone from images (a small CNN on
  84 px `agentview_image` instead of the 12-D features) — expect BC to need
  10–50× more data and to be far less sample-efficient: the classic
  state-vs-vision trade-off.
- Replace the MLP with DAgger: roll out the policy, query the *expert* for
  the correct action in the visited states, add them to the dataset,
  retrain. Does closed-loop success improve over E2's small-data BC?
  (DAgger needs an expert you can query at arbitrary states — you have one,
  it is a function!)
- AIRL with a value head and reward/value disentanglement (read the AIRL
  paper, §4) — does the learned reward transfer to a *different* cube
  placement distribution?

## 8. References

- robosuite docs: demonstration collection & robomimic dataset format
  (`robosuite.github.io`, `robomimic.github.io/docs/datasets/overview.html`)
- Ross, Gordon, Bagnell — *A Reduction of Imitation Learning ... (DAgger)*, 2011
- Ho & Ermon — *Generative Adversarial Imitation Learning*, 2016
- Finn, Christiano, Abbeel — *A Connection between GAIL and AIRL*, 2016
- BC survey: Pardo & Tadija — *Behavioral Cloning is Robust...*, or the
  classic Bojarski et al. (DAVE-2) for the "clone a controller" framing
