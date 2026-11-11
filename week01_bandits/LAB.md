# COMP0037 — Week 01 Lab: MDPs & Exploration (Bandits)

**Topic:** Markov Decision Processes, k-armed bandits, and exploration strategies (ε-greedy vs UCB).
**Folder:** `week01_bandits/` · **Files:** `mdp_formulation.py` (T1–T3), `bandits.py` (T4–T7) · **Env:** `pixi run -e week01 ...`

---

## 1. Overview

Two small systems, one idea: **before you can control anything, you must formalise the problem and decide how to explore it.**

1. **Part A — the warehouse MDP.** A robot in a 6×10 gridworld (`.` floor, `#` shelves, `m` mud, `R` start, `G` goal). Actions are U/D/L/R; with probability `slip_prob=0.2` the robot moves *perpendicular* to the intended direction instead. Rewards: `+1.0` at the goal, `-0.2` on mud, `-0.04` per step; `γ = 0.95`. You implement the reward function, the stochastic dynamics, and a Monte-Carlo estimator of a policy's expected return.

2. **Part B — the 10-armed Bernoulli bandit.** Ten actions with hidden success probabilities

   ```python
   BANDIT_PROBS = [0.10, 0.35, 0.60, 0.45, 0.80, 0.25, 0.90, 0.50, 0.70, 0.15]
   ```

   The best arm is **#6 (p\* = 0.90)**. You implement ε-greedy, incremental-mean updates, and UCB; we measure **regret** — how much reward exploration cost you versus always playing the best arm.

3. **Part C — vectorised exploration.** The same ε-greedy experiment re-implemented with array operations over `n_runs` bandit problems *simultaneously*, written so the identical code runs on **NumPy (CPU) or CuPy (GPU)**. This is the "GPU-ready NumPy" pattern used throughout the module: plain Python loops on small data, array-module code when it pays.

### Learning outcomes

After this lab you can: (a) formalise a task as an MDP ⟨S, A, T, R, γ⟩; (b) implement stochastic dynamics and a Monte-Carlo value estimator; (c) implement and *tune* ε-greedy, optimistic initialisation, and UCB; (d) explain the exploration–exploitation trade-off with regret curves; (e) vectorise an experiment and reason about when GPU batching helps.

---

## 2. Setup

**Locally (pixi, recommended).** From the `labs/` folder:

```bash
pixi run -e week01 python week01_bandits/mdp_formulation.py
pixi run -e week01 python week01_bandits/bandits.py
pixi run -e week01 python week01_bandits/bandits.py --bench          # CPU timing
pixi run -e gpu   python week01_bandits/bandits.py --bench --gpu     # CuPy timing (needs CUDA)
```

The first command creates the `week01` environment (NumPy + Matplotlib). Nothing else is needed. `bandits.py` flags: `--steps N`, `--runs N`, `--bench`, `--gpu`, `--show`.

**Google Colab (optional).** New notebook → upload the `week01_bandits/` folder (folder icon → upload; keep the file names). Then:

```python
!pip install -q numpy matplotlib
# optional, for Part C on the T4 GPU:
# !pip install -q cupy-cuda12x

%run week01_bandits/mdp_formulation.py
%run week01_bandits/bandits.py
```

Colab's default runtime (CPU) is enough; for the CuPy timing pick *Runtime → Change runtime type → T4 GPU* and install `cupy-cuda12x`. Expect the GPU to **lose** on small batches — that's the point of the benchmark (see Q7).

---

## 3. Your tasks (TODOs)

| ID | File | Function | You implement |
|----|------|----------|---------------|
| T1 | `mdp_formulation.py` | `WarehouseRobot.reward` | reward for a transition (goal/mud/step) |
| T2 | `mdp_formulation.py` | `WarehouseRobot.step` | stochastic dynamics incl. the slip rule |
| T3 | `mdp_formulation.py` | `expected_return` | Monte-Carlo estimate of V^π via `simulate_episode` |
| T4 | `bandits.py` | `EpsilonGreedy.select_action` | ε-greedy action choice |
| T5 | `bandits.py` | `BanditAgent.update` | incremental-mean Q(a) update |
| T6 | `bandits.py` | `UCB.select_action` | upper-confidence-bound rule |
| T7 | `bandits.py` | `run_vectorized_epsilon_greedy` | the whole experiment, vectorised over runs |

Every TODO raises `NotImplementedError` with a pointer back here. Read the docstrings — they contain the exact update formulas. **Read the COMPLETE code too** (`simulate_episode`, `run_experiment`, `get_xp`): you will re-use these patterns all term.

---

## 4. Experiments

Run, in order, and **keep the console output for your report**:

1. `python mdp_formulation.py` — prints the map, one episode, and expected returns for the random and hand-crafted policies.
2. `python bandits.py` — 8 agents × 1000 steps × 20 runs; saves `regret_comparison.png` and `optimal_action.png`.
3. `python bandits.py --steps 200` and `--steps 5000` — does the ranking change with horizon?
4. `python bandits.py --bench` (and `--gpu` if available) — throughput at `n_runs=20` vs `n_runs=5000`.

---

## 5. Self-check (reference numbers)

These are the numbers produced by a correct implementation with the **default seeds** (episode seeds `seed+i`; bandit base seed 0; the hint-literal RNG conventions). If your exact values differ slightly, check your RNG-draw order — but the *pattern* should match closely.

**Part A** (`mdp_formulation.py`):

```
start=(5, 0)  goal=(5, 8)  |S|=56  |A|=4  slip_prob=0.2  gamma=0.95
one episode (hand-crafted): steps=10  total reward=0.32  discounted return G=0.09
expected return of random policy:       -0.827
expected return of hand-crafted policy: +0.179
```

**Part B** (`bandits.py`, 1000 steps, 20 runs — regret @ t=1000 / % optimal actions @ t=1000):

| agent | cumulative regret | % optimal |
|---|---|---|
| ε = 0.0 (greedy) | 800.00 | 0.0% |
| ε = 0.01 | 341.33 | 30.0% |
| ε = 0.1 | 93.69 | 80.0% |
| ε = 0.5 | 218.53 | 65.0% |
| UCB, c = 0.5 | 29.93 | 100.0% |
| UCB, c = 1.0 | 95.36 | 70.0% |
| UCB, c = 2.0 | 199.92 | 65.0% |
| optimistic (Q₀ = 5) | 33.29 | 75.0% |

**Part C** (T7, ε=0.1, 20 runs × 1000 steps, NumPy): mean cumulative regret @ t=1000 ≈ **94.6** — deliberately *not* identical to the loop-based 93.69; see Q8. Wall-clock: Part A ≈ 1 s, Part B ≈ 4 s on a typical laptop.

---

## 6. Report (deliverables & questions)

Write a short report (≤ 6 pages, PDF) with:

**Deliverables**
- [ ] Your completed `mdp_formulation.py` and `bandits.py` (TODOs filled; no other changes).
- [ ] `regret_comparison.png` and `optimal_action.png`, each referenced and discussed in the text.
- [ ] Console output of the four experiment runs (append as listing or screenshots).
- [ ] Answers to Q1–Q8 below.
- [ ] **Reproducibility block:** the exact commands you ran, the seeds used (defaults: MDP `seed=0`, bandits base seed 0), Python/NumPy versions, and whether you ran locally or on Colab.

**Questions**
- **Q1 — MDP.** Write the warehouse down as ⟨S, A, T, R, γ⟩ with actual sizes (|S|=56, |A|=4). Why does the slip make T stochastic, and why does the robot's *cell* still satisfy the Markov property?
- **Q2 — Values.** Why does the hand-crafted policy beat the random one (+0.179 vs −0.827)? Give the hand-crafted policy's *optimal* expected return bound and explain why the slip and γ=0.95 keep the measured value well below the naive "+1".
- **Q3 — Greedy = 800.00.** Explain, in one paragraph, why pure greedy accumulates *exactly* 1000 × 0.80 = 800 regret in our run (hint: what does `np.argmax` do on ties, and which arm is index 0?).
- **Q4 — ε sweep.** Why does ε=0.1 dominate ε=0.01 and ε=0.5 at this horizon? What happens to each as the horizon grows (check with `--steps 5000`)?
- **Q5 — UCB.** Why does UCB with c=0.5 reach 100% optimal actions, and why do larger c values *hurt* here? What does the confidence bonus trade off?
- **Q6 — Optimism.** Why does Q₀=5 with ε=0 explore systematically rather than randomly, and why does it stop exploring? Compare its curve to ε-greedy's on `optimal_action.png`.
- **Q7 — Bench.** From your `--bench` runs: at which batch size does the GPU break even/win? Explain with per-step kernel-launch overhead vs per-element work. Report your own pulls/second numbers.
- **Q8 — Vectorised.** Your T7 result (≈94.6) differs from the loop-based ε-greedy (≈93.69) with the *same* ε and horizon. Explain why (RNG consumption order: pre-sampled reward matrix vs sequential draws), and why both are unbiased estimates of the same quantity.

**Report rules (apply every week):** every claim you make about a plot must be verifiable by looking at it; state seeds and commands; if a run failed or you deviated from the reference, say so and explain what you observed.

---

## 7. Going further (optional, not graded)

- Implement **Thompson sampling** for Bernoulli bandits (Beta posteriors) and compare with UCB.
- Make the bandit **non-stationary** (drift the probabilities every 200 steps): which agent adapts? (You will need a step-size like `α=0.1` instead of 1/n — connect this to Week 02.)
- On Colab GPU, sweep `--runs` from 20 to 200,000 and plot throughput vs batch — where does the crossover sit?

## 8. References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), §2–§2.7 (bandits), §3–§3.6 (MDPs), §4.2 (Monte-Carlo evaluation).
- Auer, Cesa-Bianchi & Fischer (2002), *Finite-time Analysis of the Multiarmed Bandit Problem* — the UCB1 bound.
- CuPy docs — `numpy`-compatible GPU arrays: https://docs.cupy.dev
