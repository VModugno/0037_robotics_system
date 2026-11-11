# Lab 07 — LLMs in Robotics: Task Planning on a Robot Skill API

**Week 07 topic:** LLMs in robotics — task planning, skill grounding, Eureka-style reward design (the reward part is "going further").

## 1. Overview

Large language models do not control robots; they *decompose tasks*. The
dominant pattern in the field (SayCan, Code-as-Policies, inner monologue,
RT-X skill hierarchies) is a two-layer system:

```
"pick up the cube"  ──LLM──▶  plan = [skill, skill, ...]  ──executor──▶  robot
                                   JSON, validated, closed-loop
```

* the **LLM** is the high-level planner: text in, structured plan out;
* the **skill API** (`robot_api.py`) is the grounding layer: a small, typed,
  safety-checkable set of closed-loop controllers on the robosuite Lift task
  (the same Panda arm and P-controller you built in week 06);
* the **planner harness** (`llm_planner.py`) is everything between them:
  prompting, JSON extraction, validation, execution.

You will write all four stages of the harness (T1–T4). The LLM itself is
**provider-agnostic and swappable** — an offline keyword mock, any
OpenAI-compatible endpoint, or a local Ollama server. The mock makes the lab
gradeable offline and, more importantly, shows that *the robotics
engineering is everything except the model*.

### Outcomes

After this lab you can:

1. design a robot skill API as a machine-readable tool schema;
2. build LLM prompts that produce machine-checkable structured output;
3. write robust JSON extraction and full-plan validation (schema + workspace
   safety) — the first two layers of the safety stack week 09 completes;
4. execute plans with stop-on-error semantics and report on failure modes.

## 2. Setup

From `labs/`:

```
pixi run --manifest-path pixi.toml -e week07 python week07_llm_planning/robot_api.py --demo
pixi run --manifest-path pixi.toml -e week07 python week07_llm_planning/llm_planner.py --provider mock --dry-run
```

`requirements.txt` (for pip/Colab): `numpy>=1.26,<2.0`, `robosuite==1.5.2`,
`mujoco>=3.3,<3.5`. **No GPU, no torch needed** — a plain CPU runtime works.

### Colab (CPU runtime is enough)

```python
# 1. upload the week07_llm_planning folder, then:
%cd week07_llm_planning
!pip install -q -r requirements.txt

# 2. the skill API, by hand:
!python robot_api.py --demo

# 3. plan + validate without a robot:
!python llm_planner.py --provider mock --dry-run

# 4. the full pipeline on the simulated robot:
!python llm_planner.py --provider mock --task "pick up the cube"
```

Notes: the env is camera-less and headless (no `MUJOCO_GL` needed).
On Windows the scripts auto-create `C:\tmp` (robosuite's offscreen
convention); robosuite's import warnings (macros file, `robosuite_models`,
mink) are benign for the Panda.

### Using a real LLM (optional but recommended)

```bash
export LLM_API_KEY=sk-...          # or leave empty for a local server
export LLM_MODEL=gpt-4o-mini      # any model your endpoint serves
export LLM_BASE_URL=https://api.openai.com/v1
pixi run --manifest-path pixi.toml -e week07 python week07_llm_planning/llm_planner.py \
    --provider openai --task "grasp the cube and lift it high"
# or a local Ollama server:
... --provider ollama --model llama3.1 --base-url http://localhost:11434
```

No SDK is used — the providers are ~30 lines of stdlib `urllib` each, so you
can read exactly what a chat-completions call is.

## 3. Reading the code, then filling the TODOs

Read `robot_api.py` first — it is **complete** (no TODOs). Understand:

* `SKILLS` — the registry: `{name: {doc, args, method}}`. This *is* the tool
  schema. T1 builds the prompt from it; T3 validates against it.
* `_drive_to` — the closed-loop P-controller with a **workspace refusal**:
  targets outside `WORKSPACE_LIMITS` are rejected up front. Notice this is
  the seed of week 09's CBF safety filter.
* skills fail loudly: `("ok", info)` or `("error", reason)`.

Then implement in `llm_planner.py`:

| TODO | Function | What you build |
|------|----------|----------------|
| T1 | `build_messages(task, state)` | system prompt: robot description + **every skill from the registry** (loop over `SKILLS`, do not hardcode) + exact output contract `{"plan": [...]}`; user prompt: state + task |
| T2 | `extract_json(text)` | first `{` to last `}`; `ValueError` if parsing fails; `None` if no braces |
| T3 | `validate_plan(plan, api)` | per step: is a dict with `skill`+`args`; skill known; args exactly match the declared set; `move_to` coords inside `api.WORKSPACE_LIMITS`. **Collect all errors with step indices.** |
| T4 | `execute_plan(api, plan)` | call each skill via `call_skill`, print results, **stop on error** returning `(False, i)`, break early on `api.state()["success"]`, return `(True, n)` at the end |

## 4. Experiments

Run and record (console output or screenshots):

* **E1 — dry run:** `--dry-run` prints state, plan, validation, and skips
  execution. Fastest way to iterate on T1–T3 without waiting for the robot.
* **E2 — mock end-to-end:** `--provider mock --task "pick up the cube"`.
  Expect `TASK SUCCESS` (see §5).
* **E3 — the API alone:** `robot_api.py --demo` (the hand-written plan) and
  `--list-skills`.
* **E4 — mock failure mode:** `--task "tidy the table"`. The mock emits its
  deliberately-useless fallback plan. Observe: valid JSON, passes validation,
  executes cleanly — and accomplishes nothing. This is the *quality* gap a
  real LLM (or a better prompt) is supposed to close. Report on it.
* **E5 — validation stress test:** craft plans (edit the mock or feed
  `--dry-run`) with an unknown skill, a missing arg, an out-of-bounds
  `move_to`, and all three at once. Verify every error is reported (T3 must
  not stop at the first).
* **E6 — real LLM (optional):** swap to `--provider openai|ollama` and try
  at least three phrasings of the task, e.g. "pick up the cube",
  "grasp the cube and lift it high", "hover above the cube". Compare plans,
  robustness of the JSON, and success rates against the mock.

## 5. Self-check

Reference run (pixi week07 env, Windows, CPU; seeds fixed by `--seed 0`):

```
E2 mock end-to-end (seed 0):
-- world state: {'eef_pos': [-0.118, 0.002, 0.997], 'cube_pos': [-0.01, 0.015, 0.831], ...}
[plan 0] move_above  {'target': 'cube', 'dz': 0.12} -> ok {'steps': 10}
[plan 1] descend_to_cube {'dz': 0.0}                 -> ok {'steps': 11}
[plan 2] close_gripper {}                           -> ok {'gripper': 'closed'}
[plan 3] lift {'height': 0.15}                      -> ok {'steps': 13}
[i] task succeeded early -- stopping after step 3
-- final state: {'cube_pos': [-0.008, 0.015, 0.966], 'success': True}
TASK SUCCESS
```

* the whole pipeline (env init + 4 skills) runs in well under a minute on CPU;
* validation stress test: a plan with unknown skill + missing arg +
  out-of-bounds `move_to` + extra arg + non-dict step yields **five** errors
  in one pass, each with its step index;
* `--dry-run` exits 0 without ever constructing the robosuite env;
* exact step counts and cube positions will differ slightly on your machine
  (robosuite seeds placement from numpy at env construction) — the
  qualitative result (4 skills, early success stop, cube ≈ 0.13–0.15 m above
  the table) is what should reproduce.

If `TASK FAILED` appears: check the executor printout — the usual causes are
a `descend_to_cube` target at the wrong height (T3 should not have caught
it; T2 might have mis-parsed) or a plan that closes the gripper too early.

## 6. Report

Deliverables (checklist):

- [ ] console output of E1–E5 (and E6 if attempted), annotated;
- [ ] the exact system prompt you built in T1 (quote it) and a short
      justification per section (why list args, why the output contract);
- [ ] the list of validation errors E5 produced;
- [ ] your answers to Q1–Q6 below;
- [ ] reproducibility: commands, seeds, provider + model used.

Questions:

* **Q1 — prompt anatomy.** Which parts of your T1 prompt are load-bearing?
  Justify with evidence: run the mock (it ignores the prompt! it keyword
  matches) versus a real LLM with an ablated prompt (drop the output
  contract, or the arg types, or the state). Which ablations break parsing,
  which break planning quality?
* **Q2 — parsing robustness.** List three ways real LLMs wrap JSON (prose,
  fences, trailing commentary) and show `extract_json` surviving each (feed
  it strings directly in a REPL). What does it do with text that has no
  braces — and why is returning `None` (not raising) the better contract
  for the driver in `plan_task`?
* **Q3 — validation as a safety layer.** T3 refuses out-of-workspace
  targets *before* execution. Compare this to week 06 (the expert's implicit
  safety) and week 04 (the CBF shield): what does each layer catch that the
  others cannot? Where does this stack end up in week 09?
* **Q4 — closed-loop skills, open-loop plan.** The plan is a fixed sequence,
  yet the cube's exact position never enters it. Explain why the plan still
  works (hint: `skill_move_above` reads the *live* cube position — the same
  live-tracking fix as the week 06 expert). What class of disturbances
  would still break a fixed plan?
* **Q5 — stop vs replan.** The executor stops at the first skill error.
  Describe (in 5–10 sentences, no code needed) how you would extend it to
  replan: what state goes back into the prompt, what the LLM needs to know
  about the failure, and what the new failure modes of *that* design are.
* **Q6 — mock vs model.** The mock's fallback plan is "plausible but
  useless" — valid JSON, valid skills, zero task progress. Argue which of
  the three levers (better model, better prompt, stricter validation) fixes
  which failure, using your E4/E6 evidence.

Reproducibility: pin `--seed`, quote the provider and model, and paste the
exact commands. If you used a real LLM, note the temperature (the harness
fixes it at 0.0 — why?).

## 7. Going further

* **Eureka-style reward design:** instead of a plan, ask an LLM to *write a
  reward function* for the Lift task (a Python function of the state),
  inject it into a small RL loop, and let it propose refinements from the
  return curve. This is the Eureka loop from the lecture.
* **State feedback into the prompt:** after each skill, append the observed
  state to the conversation and let the LLM re-plan (inner-monologue
  style). You already have all the pieces.
* **Affordance scoring (SayCan):** have the LLM output skill *candidates*
  with probabilities and multiply by a learned value per skill — you have a
  BC policy from week 06 that could supply that value.
* **Structured-output APIs:** compare your hand-rolled JSON contract with a
  provider's native function-calling / JSON-mode. Same interface, different
  guarantees.

## 8. References

* robosuite (documentation for the Lift task and Panda) — https://robosuite.ai
* SayCan: Ahn et al., *Do As I Can, Not As I Say* (2022)
* Code as Policies: Liang et al. (2023)
* Inner Monologue: Huang et al. (2022)
* Eureka: Ma et al., *Human-Level Reward Design via Coding Large Language
  Models* (2023)
