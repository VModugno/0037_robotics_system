# COMP0037 — Week 04 Lab: Continuous Control & Safe RL (Actor-Critic + CBF Shield)

**Topic:** Policy-gradient methods for continuous action spaces, GAE, and control-barrier-function safety filters.
**Folder:** `week04_actor_critic/` · **Files:** `arm2d_env.py` (COMPLETE), `actor_critic.py` (T1–T4) · **Env:** `pixi run -e week04 ...`

---

## 1. Overview

Q-functions struggle with continuous actions (that max over a′ from Week 03 becomes an optimisation problem). This week you *parameterise the policy itself* — a neural network outputs the mean of a Gaussian over joint velocities — and learn it from the policy-gradient theorem. A critic (state-value network) converts returns into advantages; **GAE** balances their bias and variance.

Then the week's twist: **safety**. The arm has hard joint limits (±2.6 rad). The unconstrained learner happily slams into them — targets are sampled up to 0.7 m while the arm reaches 0.8 m, so many optimal solutions live near the workspace boundary. Instead of baking constraints into the reward, we attach a **control barrier function (CBF) shield**: a one-sided filter that intervenes only when an action would push a joint past its limit *this step*, clipping it in closed form. The agent is free; the shield guarantees.

`main()` trains the SAME agent twice — unshielded, then shielded — so you can measure exactly what safety costs (and here: whether it costs anything at all).

### Learning outcomes

(a) Implement Gaussian-policy sampling and the actor-critic loss; (b) implement GAE and explain λ; (c) derive and implement a discrete-time CBF-style safety filter; (d) quantify safety vs return trade-offs empirically; (e) reason about evaluating stochastic policies.

---

## 2. Setup

**Locally (pixi).** From `labs/`:

```bash
pixi run -e week04 python week04_actor_critic/arm2d_env.py     # self-test (run FIRST)
pixi run -e week04 python week04_actor_critic/actor_critic.py   # both configs, ~50 s CPU
```

Flags: `--episodes N` (default 400), `--device cpu|cuda|auto`, `--show`. The run trains twice (unsafe, then shielded) and writes `a2c_comparison.png` (3 panels: returns, cumulative limit hits, greedy success).

**Google Colab.** Upload `week04_actor_critic/`:

```python
!pip install -q torch numpy matplotlib

%run week04_actor_critic/arm2d_env.py
%run week04_actor_critic/actor_critic.py
```

CPU is fine; a GPU runtime changes little at this scale (say so in your report if you tried).

---

## 3. Your tasks (TODOs)

| ID | File | Function | You implement |
|----|------|----------|---------------|
| T1 | `actor_critic.py` | `GaussianActor.sample_action` | sample from the Normal, return a (2,) numpy action |
| T2 | `actor_critic.py` | `compute_gae` | δ_t = r + γV(s′) − V(s); A_t = δ_t + γλ·A_{t+1} (backwards); returns = A + V |
| T3 | `actor_critic.py` | `cbf_safe_action` | one-sided joint-limit clip (derivation in the docstring) |
| T4 | `actor_critic.py` | `A2CAgent.update` | actor loss = −(log π·Â).mean() − c_ent·entropy + critic MSE |

Read `arm2d_env.py` first: it defines the action convention (a ∈ [−1,1]² → ω = a·OMEGA_MAX, θ += ω·DT), the reward (−dist − 0.01‖ω‖², +1 at success), and the **limit_hit** flag that the unsafe run will accumulate. `safety_shield` (COMPLETE) wraps your T3 and handles the unit conversion — check that you understand the round-trip action→ω→action before implementing T3.

**T2 note:** the docstring recursion does not use the `dones` argument — terminality is already encoded by the caller in `next_values` (success ends the episode, so `done` can only occur at the last index). The `(1−done)` masking version gives numerically identical results *here*; we discuss why in Q2.

---

## 4. Experiments

1. **Self-test** (`arm2d_env.py`): checks hard clamping at the joint limits with a random policy.
2. **Main run** (`actor_critic.py`, 400 episodes × 2 configs): record for BOTH configs the last-20 return, cumulative joint-limit hits, and the greedy success trend.
3. **Episode budget:** re-run with `--episodes 150` and compare: is the shield's benefit already visible early?
4. Optional: raise `h_min` in T3 from 0.05 to 0.3 rad (more conservative margin) and re-run — what happens to returns and hits?

---

## 5. Self-check (reference numbers)

Reference run: 400 episodes, CPU, **seed 0** (set in `main()`, so your run should match closely).

**Self-test:**

```
random uniform policy: successes=1/10, joint-limit clamps=5
joint-limit clamping verified.
```

**Main run (400 episodes, device=cpu):**

| config | return (last-20 mean) | cumulative limit hits | hits/ep (last-20) | final greedy success | wall |
|---|---|---|---|---|---|
| no shield (unsafe) | **−112.45** | **31,433** | 133.2 | 0% (0/10) | 21 s |
| CBF shield (safe) | **−96.49** | **0** | 0.0 | 0% (0/10) | 25 s |

Training dynamics to expect (per-25-episode eval lines):

- **Unsafe:** return hovers around −86…−120; `limit hits/ep` *grows* from 0.8 to ~133 — the policy increasingly **parks against the joint limits** because many targets sit near the workspace boundary. A 20% greedy blip can appear mid-training; final greedy evals are 0/10.
- **Shielded:** zero clamps from the first episode to the last; return ends slightly **better** (−96 vs −112); greedy success transiently reaches 20–30% around episodes 50–150, then decays to 0/10 at the end.

The shield guarantees |θ_{t+1}| ≤ limit − h_min whenever it intervenes — with `h_min = 0.05` and limits ±2.6, a shielded policy *cannot* clamp. If your unsafe/shielded hits don't look like 31,433 vs **exactly 0**, re-check T3's one-sidedness (it must never clip inward-moving actions).

Wall-clock ≈ 50 s total on one CPU core.

---

## 6. Report (deliverables & questions)

**Deliverables**
- [ ] Completed `actor_critic.py`.
- [ ] `a2c_comparison.png` with a caption per panel.
- [ ] Console logs of both runs (400 eps + your 150-eps comparison).
- [ ] Answers to Q1–Q6.
- [ ] **Reproducibility block:** commands, seed (default 0 — `main()` calls `torch.manual_seed(0)`), versions, hardware.

**Questions**
- **Q1 — Anatomy.** Why two networks? Which loss term trains the critic, and why does the actor need the critic at all (compare with REINFORCE's ∇log π·G variance)? Explain the negative sign and the detached advantage in T4.
- **Q2 — GAE.** Derive A_t from the δ recursion and explain the role of λ (what do λ=0 and λ=1 reduce to? bias/variance trade-off). Then: why does ignoring the `dones` mask here still give the same result, and under what episode structure would that break?
- **Q3 — Entropy.** What does the −c·H(π) term prevent? Predict (then check) what the unsafe run's hits/ep trend would look like with `entropy_coef = 0`.
- **Q4 — The shield.** Reproduce the one-line derivation: from |θ + ω·dt| ≤ limit − h_min, solve for the max |ω| that keeps a joint safe *only when the action pushes it outward* (θ·ω > 0). Why is the filter one-sided? What is `h_min` for, and what happens as it grows (your Experiment 4)?
- **Q5 — Why park?** Explain the unsafe run's 0.8 → 133 hits/ep growth: connect the target distribution (radius up to 0.7 m) to the arm's reach (0.8 m) and the per-step reward shape. Why is limit-parking rational for the unconstrained learner?
- **Q6 — Free safety?** The shielded run ends with *better* return (−96 vs −112) and zero clamps. Give at least two mechanisms (e.g., the smoothness prior imposed near limits; the −0.01‖ω‖² action cost interacting with saturation; evaluation noise). Then discuss the greedy success numbers: both configs end at 0/10 — what does that say about evaluating a *stochastic* training regime with *greedy* rollouts, and which metric (return? hits? success?) would you defend as the primary one here and why?

**Report rules:** as before — claims checkable against plots/logs, seeds and commands stated, deviations reported honestly.

---

## 7. Going further (optional)

- Swap the A2C update for **PPO**'s clipped surrogate (ratio clamp) — you already have logp; ~10 lines.
- Make the shield softer: instead of hard clipping, add a CBF-style barrier term QP (the `qpsolvers` package arrives with robosuite in Week 06).
- Track the *distance-to-limit* margin over training and plot it per config — visualise the shield's "don't get close" effect vs the unsafe policy's distribution piling up against ±2.6 rad.

## 8. References

- Sutton & Barto (2nd ed.), §13.1–13.5 (policy gradients, actor-critic).
- Schulman et al. (2016), *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (GAE).
- Ames et al. (2019), *Control Barrier Functions: Theory and Applications* — the safety-filter perspective used here.
