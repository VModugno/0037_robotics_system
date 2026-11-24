# COMP0037 — Week 02 Lab: Tabular Model-Free RL (TD Learning)

**Topic:** Temporal-difference learning, Q-Learning vs Sarsa, and the exploration–safety interaction.
**Folder:** `week02_tabular_rl/` · **Files:** `gridworld.py` (COMPLETE), `agents.py` (T1–T3), `run_lab.py` (driver) · **Env:** `pixi run -e week02 ...`

---

## 1. Overview

Last week you formalised an MDP and estimated returns by Monte-Carlo roll-outs. This week you learn **on-line, from single steps, without a model**: temporal-difference (TD) learning.

Two update rules, one line apart, with opposite personalities:

```
Q-Learning (off-policy):  Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]
Sarsa (on-policy):         Q(s,a) ← Q(s,a) + α [ r + γ ·      Q(s',a') − Q(s,a) ]
```

Q-Learning bootstraps from the **greedy** successor value; Sarsa from the value of the action its **ε-greedy behaviour policy** will actually take next. On safe environments they agree. Near a cliff they do not.

You implement the two rules on:

- **WarehouseEnv** — the Week-01 warehouse, stochastic (slip 0.2), γ=0.95: `−0.04/step, −0.2 mud, +1.0 goal`.
- **CliffEnv** — Sutton & Barto's cliff walking (4×12, deterministic, γ=1.0): `−1/step, −100 + reset on the cliff, 0 on the goal step`.

### Learning outcomes

(a) Implement ε-greedy behaviour and the two TD updates from their definitions; (b) explain on- vs off-policy learning *empirically*; (c) read value functions off a rendered policy; (d) study the effect of the step size α; (e) recognise the classic Q-Learning "cliff" pathology and explain its cause.

---

## 2. Setup

**Locally (pixi).** From the `labs/` folder:

```bash
pixi run -e week02 python week02_tabular_rl/gridworld.py   # env self-test (run this first)
pixi run -e week02 python week02_tabular_rl/run_lab.py     # all three experiments
```

The pixi environments set `PYTHONUTF8=1`, so the Unicode policy arrows print fine on Windows. If you run the files with a *bare* Python instead, use `python -X utf8 run_lab.py` (Windows consoles default to cp1252 and would otherwise crash on the arrows).

**Google Colab.** Upload the `week02_tabular_rl/` folder, then:

```python
!pip install -q numpy matplotlib

%run week02_tabular_rl/gridworld.py
%run week02_tabular_rl/run_lab.py
```

CPU runtime is plenty — the full lab runs in ~3 seconds.

---

## 3. Your tasks (TODOs)

All in `agents.py` (the environments and the driver are COMPLETE — but read them: `gridworld.py` shows a clean tabular env interface you will reuse in Week 03):

| ID | Function | You implement |
|----|----------|---------------|
| T1 | `select_epsilon_greedy(Q, s_idx, eps, rng)` | ε-greedy action selection (random action with prob ε, else `argmax Q[s_idx]`) |
| T2 | `q_learning_update(...)` | the off-policy TD target (max over next actions; target = r when done) |
| T3 | `sarsa_update(...)` | the on-policy TD target (uses `a2`; target = r when done) |

The driver `run_lab.py` calls `train(env, method, ...)` from `agents.py` with both `"q_learning"` and `"sarsa"`; `train` is COMPLETE — read its loop carefully: it draws `a2` even for Q-Learning (it needs it for Sarsa) and passes `a2 = -1` as a **sentinel** when the episode ended. Mind that sentinel in T3 (see Q5).

---

## 4. Experiments

`run_lab.py` runs all three automatically (~3 s total). Keep the output:

1. **Exp 1 — WarehouseEnv** (600 episodes, α=0.2, γ=0.95, ε=0.1, seed 1): learning curves for both methods → `warehouse_learning.png`.
2. **Exp 2 — CliffEnv** (500 episodes, α=0.5, γ=1.0, ε=0.1, seed 2): learning curves + side-by-side ASCII renders of the two greedy policies → `cliff_learning.png`.
3. **Exp 3 — α sweep** (CliffEnv, Q-Learning, 300 episodes, α ∈ {0.1, 0.3, 0.5, 0.7, 1.0}, seed 3): printed table of final returns.

Then re-run Exp 2 with `epsilon=0.01` and `epsilon=0.25` by editing the call in `run_lab.py` (one line) and observe how the gap between the two methods responds.

---

## 5. Self-check (reference numbers)

Correct implementations with the built-in seeds (policy RNG `default_rng(seed+7)`, env re-seeded per run) give:

**Env self-test** (`gridworld.py`):

```
WarehouseEnv: |S|=56, random walk 130 steps, total reward -5.0
CliffEnv: |S|=48, random walk 200 steps, total reward -3269.0
```

**Exp 1 (warehouse, last-100-episode mean return):**

```
q_learning: +0.464
     sarsa: +0.422
```

**Exp 2 (cliff, last-100-episode mean return):**

```
q_learning: -59.8        ← optimal target, risky behaviour
     sarsa: -22.9        ← safer path, better behaviour return
```

The rendered greedy policies: **Q-Learning takes the optimal 13-step path along the cliff edge** (optimal return −12: twelve −1 steps plus the free goal step); **Sarsa routes along the top row** (17 steps, return −16).

**Exp 3 (α sweep, Q-Learning on the cliff):**

| α | last-100 mean return |
|---|---|
| 0.1 | −51.8 |
| 0.3 | −36.1 |
| 0.5 | −37.8 |
| 0.7 | −43.6 |
| 1.0 | −42.0 |

*Exact* numbers depend on the order in which your T1 consumes RNG draws (our reference draws `rng.random()` every step, then `rng.integers()` only on explore; `argmax` ties break to the first index, i.e. UP). If your values are within a few units of the table above and show the same pattern, you are done — record any deviation in your notes.

Wall-clock: ~2.3 s for the whole lab on a typical laptop.

---

## 6. Deliverables & Discussion Questions

Record your experimental findings and answer the following questions in your lab notes:

**Deliverables checklist**
- [ ] Your completed `agents.py`.
- [ ] `warehouse_learning.png` and `cliff_learning.png` saved in `week02_tabular_rl/`.
- [ ] The α-sweep table (yours) plus the two policy ASCII renders from Exp 2.
- [ ] Answers to Q1–Q6 below.
- [ ] **Reproducibility block:** commands, seeds (defaults: 1, 2, 3), library versions, local vs Colab.

**Questions**
- **Q1 — Targets.** State the Q-Learning and Sarsa targets you implemented and label each as on-policy or off-policy. What exactly does "off-policy" mean here — what two policies appear inside one Q-Learning update?
- **Q2 — The cliff gap.** Why does Sarsa's *behaviour* return (−22.9) beat Q-Learning's (−59.8) while Q-Learning's *greedy* policy is the better one (−12 vs −16)? Use the ε=0.1 exploration term in your explanation, and support it with your ε=0.01 / ε=0.25 re-runs: how does the gap change, and why?
- **Q3 — Reading policies.** In your Exp 2 render, why does Sarsa's *greedy* policy avoid the cliff edge even though the CliffEnv is deterministic (a greedy walk can never slip)? What does this tell you about where the Q-values came from?
- **Q4 — Step size.** Interpret the α sweep: why does α=0.1 underperform at 300 episodes, why is there a sweet spot around 0.3–0.5, and why is α=1.0 surprisingly competitive on this *deterministic* environment but would be a bad idea on the slippery warehouse? (Relate α=1.0 Q-Learning on a deterministic MDP to asynchronous value iteration.)
- **Q5 — The sentinel.** `train` passes `a2 = -1` when the episode is done. What exact bug would you have introduced in T3 if you forgot the `done` guard and used `Q[s2_idx, a2]`? (Check what NumPy does with a negative index.) Why does the same mistake NOT crash but silently corrupt learning?
- **Q6 — Warehouse.** Why is the Q-Learning/Sarsa difference so small on the warehouse compared to the cliff? Identify the structural features of each MDP that make risk matter.

**Scientific rigor:** as Week 01 — every plot claim checkable by eye, seeds and commands stated, deviations recorded honestly.

---

## 7. Going further (optional)

- Implement **Expected Sarsa** and place it between the two (it should match Q-Learning's target on the cliff but Sarsa's behaviour).
- Add **double Q-Learning** (two estimates, max on one, update on the other) — does the cliff behaviour return improve?
- Plot **greedy-policy value vs episode** on top of the behaviour-return curves: this is the cleanest picture of the Q-Learning "optimist's" gap.

## 8. References

- Sutton & Barto (2nd ed.), §6.1–6.4 (TD), §6.5 (Sarsa), §6.6 (Q-Learning), Example 6.6 ("cliff walking").
- Watkins & Dayan (1992), *Q-learning*, Machine Learning 8.
