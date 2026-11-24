# Week 9 Lab — System Integration: LLM Planner → Skills → CBF Safety Filter → Robot

Capstone lab: assemble the pieces from the whole module into one pipeline and
study what happens at the seams. A (mock) LLM planner maps task text to a skill
plan, a low-level skill policy drives the Panda in robosuite `Lift`, and a
control-barrier-function-style filter intercepts every commanded end-effector
target before it reaches the robot.

    task text -> [T3 planner] -> skill plan -> [T4 executor] -> skills -> robot
                                            every geometric target
                                            -> [T1/T2 safety filter]

## 1. Overview & outcomes

By the end of this lab you can:

- implement a CBF-style projection filter (hard workspace box + margin) and a
  barrier `h()` that reports how close a *commanded* motion came to each boundary;
- wire a planner → policy → filter → robot loop and instrument it (success,
  steps, interventions, near-misses per episode);
- demonstrate, with numbers, the core safety argument of the module: **the
  planner does not have to be trustworthy if the filter is** — the `slam` plan
  is blocked by the filter while the pick plan passes untouched;
- reproduce a clean comparison run (filter ON vs OFF) and report it.

This is a mini-project: the TODO surface is small (T1–T5), but you own the
experiment design, the measurement and the write-up.

## 2. Setup

From the `labs/` root (pixi ≥ 0.80):

```
pixi install -e week09
pixi run -e week09 python week09_integration/safety_filter.py     # self-test
pixi run -e week09 python week09_integration/pipeline.py --episodes 5
```

Or from inside `week09_integration/`:

```
pixi run --manifest-path ../pixi.toml -e week09 python pipeline.py --episodes 5
```

CPU-only — no torch needed for Track A. Colab: upload the folder, then

```python
!pip install "numpy>=1.26,<2.0" robosuite==1.5.2 "mujoco>=3.3,<3.5"
import os; os.environ.setdefault("MUJOCO_GL", "egl")
!python safety_filter.py
!python pipeline.py --episodes 5
```

Windows notes (same as weeks 6–8): create `C:\tmp` before importing robosuite
(the lab code does this for you), and if the MuJoCo DLL fails to load, copy
`mujoco.dll` from the `mujoco` package into `robosuite\utils` (week 6 LAB.md).

## 3. Your TODOs

| # | file | function | what you implement |
|---|------|----------|--------------------|
| T1 | `safety_filter.py` | `SafetyFilter.filter_target` | per-axis clamp into `[lower+h_min, upper−h_min]`, count interventions, return a NEW array |
| T2 | `safety_filter.py` | `SafetyFilter.h` | signed distance of the ORIGINAL command to each of the 6 walls (no clamping!) |
| T3 | `pipeline.py` | `plan_task_mock` | keyword → plan mapping, including the deliberately dangerous fallback |
| T4 | `pipeline.py` | `run_episode` | execute a plan, early-stop on success, near-miss accounting via `check_near_miss` on the original target |
| T5 | `pipeline.py` | `summarise` | markdown results table + totals line |

Everything else is given: the `RobotAPI` (P-controller, same spirit as week 7),
`execute_skill` (returns `(ok, reason, target)` — the third value is the
*unfiltered* geometric target, None for `close`/`wait`), the `PLANS` dict, and
`main` with `--task/--episodes/--unsafe/--seed/--json-out` flags.

Notes that matter:

- `execute_skill` already runs every geometric target through
  `filter_target` (that is where interventions are counted). Your T4 only adds
  the near-miss side: `flt.check_near_miss(target)` on the original target —
  a near-miss is a property of the *command*, which is exactly what a barrier
  monitors; the filter silently clamping it afterwards is a separate event.
- The per-episode counters are reset in `main` *after* each result is appended,
  so read them inside `run_episode` before returning.
- Why did the first `--unsafe` slam run in our reference session push the arm
  toward z = 0.20 for 60 steps and fail? `drive_to` caps at 60 steps; the
  physical table blocks the gripper. The filter exists precisely so the arm
  never spends those steps trying.

## 4. Experiments

- **E1 — clean pick.** `--task "pick up the cube" --episodes 5` with the filter
  ON. Expect 5/5, zero interventions, zero near-misses.
- **E2 — the slam interlock.** `--task "slam the cube"` (a plan whose `move_to`
  target is z = 0.20, deep below the safe floor). Run it twice: with the filter
  ON and with `--unsafe`. Compare the tables: what does the filter turn the
  command into, what does the episode report, and what does the unsafe arm do?
- **E3 — margin sensitivity.** Sweep `h_min` in `{0.01, 0.03, 0.05, 0.08}` by
  editing the `SafetyFilter()` construction (or add a flag). At some point the
  *pick* plan starts taking interventions too (the descent target cube-centre
  ≈ 0.82 gets clamped to `0.79 + h_min`). Find the threshold, explain it from
  the geometry, and discuss the safety-vs-capability tradeoff. This is the
  CBF lesson of the week in one plot.
- **E4 — planner fallback.** Feed a nonsense task (`--task "make me a coffee"`).
  The mock planner falls back to the dangerous plan — on purpose. Show that the
  system survives an unhelpful planner, and discuss why the fallback design is
  "dangerous-but-safe" rather than "harmless".
- **E5 — reporting.** `--json-out results.json` on your best run; the reasons
  list is your audit trail — inspect the clamp messages.

## 5. Self-check (reference run, seed 0, CPU)

Reference implementation output (all stages):

```
-- stage: selftest ----------------------------------------
safety_filter self-test OK - safe: [0.22, -0.32, 0.82]
-- stage: pick (filter ON) ----------------------------------------
task: 'pick up the cube' -> plan: ['move_above', 'descend', 'close', 'lift'] | filter: ON
| ep | success | steps | interventions | near_misses | last skill |
|----|---------|-------|---------------|-------------|------------|
| 0 | True | 36 | 0 | 0 | lift |
| 1 | True | 32 | 0 | 0 | lift |
| 2 | True | 37 | 0 | 0 | lift |
| 3 | True | 38 | 0 | 0 | lift |
| 4 | True | 34 | 0 | 0 | lift |
5/5 successes, 0 filter interventions, 0 near-misses
-- stage: slam (filter ON) ----------------------------------------
task: 'slam the cube' -> plan: ['move_to', 'wait'] | filter: ON
| ep | success | steps | interventions | near_misses | last skill |
|----|---------|-------|---------------|-------------|------------|
| 0 | False | 18 | 1 | 1 | wait |
| 1 | False | 18 | 1 | 1 | wait |
0/2 successes, 2 filter interventions, 2 near-misses
-- stage: slam (filter OFF) ----------------------------------------
task: 'slam the cube' -> plan: ['move_to', 'wait'] | filter: OFF (unsafe)
| ep | success | steps | interventions | near_misses | last skill |
|----|---------|-------|---------------|-------------|------------|
| 0 | False | 60 | 0 | 0 | move_to |
| 1 | False | 60 | 0 | 0 | move_to |
0/2 successes, 0 filter interventions, 0 near-misses
```

Episode reasons (from `--json-out`):

- pick, ep 0: `reached [0.001, 0.026, 0.951]` → `reached [0.001, 0.026, 0.821]`
  (descent to cube centre, *unclamped* — 0.821 is above the effective floor
  0.82 by 1 mm — the margin is genuinely tight, see E3) → `gripper closed` →
  `reached [0.001, 0.026, 0.971]` (lift).
- slam, ep 0 (filter ON): `reached [0.0, 0.0, 0.82]` — the commanded
  `[0, 0, 0.20]` was projected to the floor and *reached there*, then `waited 5`.
  One intervention, one near-miss (`h["z-"] = 0.20 − 0.79 = −0.59 < h_min`).
- slam (filter OFF): `MISSED [0.0, 0.0, 0.2]` after 60 steps of pushing into
  the table.

Runtime: ~40 s for the whole stage-`all` run (7 episodes + self-test) on CPU.
Exact step counts vary with the cube placement seed; the pattern
(pick clean / slam blocked / unsafe slam misses) is the invariant.

## 6. Deliverables & Discussion Questions

Record your experimental findings and answer the following questions in your lab notes:

**Deliverables checklist**

- [ ] `safety_filter.py` self-test output.
- [ ] The three comparison tables (pick ON, slam ON, slam OFF) — copy-paste
      them verbatim.
- [ ] E3 threshold: the `h_min` value where pick starts intervening, plus your
      geometric explanation (table ≈ 0.81, cube centre ≈ 0.82, floor 0.79).
- [ ] `results.json` from your best run.
- [ ] Answers to Q1–Q6 below.

Questions:

- **Q1** — This filter clamps *position targets*. Week 4's shield clipped
  *joint velocities* to keep `|θ| ≤ limit − h_min`. Both are "CBF-style":
  explain what a control barrier function is in general, and what each
  implementation preserves (state constraint vs one-step reachability).
- **Q2** — The near-miss counter fires on the *commanded* target even though
  the filter guarantees the executed target is safe. Why is monitoring the
  command the right quantity for a safety audit? What would you log in a
  real deployment (think: an operator reading this dashboard).
- **Q3** — The `slam` plan "completes" with the filter ON (last skill `wait`,
  18 steps) and "fails" with the filter OFF (`MISSED`, 60 steps). Which
  behaviour would you rather ship, and how do you reconcile "plan succeeded"
  with "task was never going to happen"?
- **Q4** — The mock planner's fallback for unknown tasks is the dangerous plan
  (deliberately). Contrast with the alternative design: fallback = `hover`.
  Discuss detectability (which design surfaces the planner's failure?) versus
  defensiveness. Connect to week 7's validation layer.
- **Q5** — From E3: at what `h_min` does the pick plan lose capability? What
  is the "right" margin for a real cell with humans nearby? What information
  would you need to choose it?
- **Q6** — System view: pick ONE failure mode from any week 1–8 lab (e.g. the
  week 6 stale-waypoint expert, week 8's undertrained diffusion head, week 7's
  plausible-but-useless mock plan) and explain how this pipeline's layering
  (planner → skills → filter) would or would not contain it.

Reproducibility block:

```
machine / OS / CPU:
pixi env: week09
commands:
  pixi run -e week09 python week09_integration/safety_filter.py
  pixi run -e week09 python week09_integration/pipeline.py --task "..." --episodes N [--unsafe] --seed 0
seed: 0
```

## 7. Going further

- **Real LLM planner.** The mock keyword planner is one `llm_planner.py`-style
  call away from week 7's providers: `LLM_API_KEY` / `LLM_BASE_URL` /
  `LLM_MODEL` env vars, JSON plan → validate against `PLANS`/skill args →
  `execute_skill` loop. Keep the filter ON and re-run E4 with a real model.
- **Diffusion skill policy.** `DiffusionPolicyAdapter` (stub in `pipeline.py`):
  load your week-8 `diffusion_head.pt` (or `diffusion_head_vlm.pt`), reuse
  week 8's `evaluate_policy` chunk-MPC loop, and wrap the resulting geometric
  target with this week's `SafetyFilter`. Now the filter constrains a *learned*
  policy — the actual week-9 scenario of a real robot stack.
- **Track B — full VLA stack.** Week 8's Foundry track as the low-level policy:
  image + instruction in, 20-D action out, sliced to OSC_POSE; the filter then
  guards *that* output. On Colab/server with a GPU (see week 8 LAB.md Track B).
- **Velocity barriers.** Extend the filter to reject commands whose *approach
  speed* toward a wall exceeds a braking profile (a true CBF on `ḣ`, not just
  `h > 0` — closer to week 4's shield).

## 8. References

- Ames et al., *Control Barrier Functions: Theory and Applications* (ECC 2019 tutorial).
- robosuite docs — `suite.make`, OSC_POSE controller conventions.
- Weeks 4 (joint-limit shield), 6 (scripted expert), 7 (LLM planner +
  validation), 8 (diffusion head) of this module.
