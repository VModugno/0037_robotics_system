# COMP0037 — Week 03 Lab: Deep Q-Networks (DQN)

**Topic:** Value-based deep RL: experience replay, target networks, and the machinery of TD learning at scale.
**Folder:** `week03_dqn/` · **Files:** `reach_env.py` (T1 + COMPLETE env), `dqn.py` (T2–T6) · **Env:** `pixi run -e week03 ...`

---

## 1. Overview

Tabular Q-Learning dies when the state stops fitting in a table. This week the state is continuous — a 2-link planar arm that must reach a random target — and the Q-function becomes a small neural network. You implement the **inner mechanics of DQN** exactly as in Mnih et al. (2015): a replay ring buffer, ε-greedy action selection, the bootstrapped TD target, and the gradient step.

**The environment (`reach_env.py`, mostly COMPLETE).** Two links (L1=0.45 m, L2=0.35 m, joint limits ±2.6 rad) must move their end-effector within 0.05 m of a target sampled at radius 0.30–0.70 m. Actions are 9 joint-velocity deltas (du, dv ∈ {−1, 0, +1}); observations are 6 floats (sin/cos of both joint angles + target x, y); episodes last 100 steps; reward = −distance, +1.0 at success. You implement the **forward kinematics** (T1) — the geometric heart of the robot.

**The agent (`dqn.py`).** `QNet` maps 6 floats → 9 Q-values. `train()` (COMPLETE — read it line by line) wires replay, target network, linear ε-decay, and periodic greedy evaluation; you implement what's inside:

### Learning outcomes

(a) Implement a replay buffer and sample minibatches as tensors; (b) implement ε-greedy selection with a network; (c) compute the DQN regression target and explain *why* it is what it is; (d) run and interpret ablations of the two stabilising tricks; (e) explain truncation vs termination — a bug class that silently breaks many RL pipelines.

---

## 2. Setup

**Locally (pixi).** From `labs/`:

```bash
pixi run -e week03 python week03_dqn/reach_env.py            # self-test (do this FIRST)
pixi run -e week03 python week03_dqn/dqn.py --episodes 300   # main run, ~80 s on CPU
pixi run -e week03 python week03_dqn/dqn.py --episodes 150 --no-target-net   # ablation 1
pixi run -e week03 python week03_dqn/dqn.py --episodes 150 --small-buffer   # ablation 2
```

First run downloads PyTorch (~200 MB one-off). The pixi environment also sets the Windows OpenMP compatibility flag, so nothing else is needed. Useful flags: `--device cpu|cuda|auto`, `--show`.

**Google Colab (T4 GPU works well).** Upload `week03_dqn/`, then:

```python
!pip install -q torch numpy matplotlib

%run week03_dqn/reach_env.py
%run week03_dqn/dqn.py --episodes 300
```

The DQN is tiny (6→128→128→9); CPU is fine, but try `--device cuda` once and record the speed difference for Q7.

---

## 3. Your tasks (TODOs)

| ID | File | Function | You implement |
|----|------|----------|---------------|
| T1 | `reach_env.py` | `forward_kinematics(θ1, θ2)` | 2-link FK: x = L1 cos θ1 + L2 cos(θ1+θ2), y likewise with sin |
| T2 | `dqn.py` | `ReplayBuffer.push` | ring-buffer write: position `pos % capacity`, size cap |
| T3 | `dqn.py` | `ReplayBuffer.sample` | uniform indices → CPU torch tensors dict (actions shaped (B,1) int64) |
| T4 | `dqn.py` | `DQNAgent.act` | ε-greedy on ONE observation (torch.no_grad, plain int) |
| T5 | `dqn.py` | `DQNAgent.td_target` | y = r + γ(1−done)·max_a' q_target(s')[a'] |
| T6 | `dqn.py` | `DQNAgent.optimize` | gather → Huber loss → clip grads (10.0) → Adam step |

Run `reach_env.py` first: its self-test checks your FK against three geometric ground truths, then rolls a random policy. `dqn.py` greets you with `NotImplementedError` until T2–T6 are done.

---

## 4. Experiments

1. **Main run** (300 episodes): watch `return(20ep)` climb and `greedy success` trend up. Produces `dqn_learning.png`, `dqn_final.pt`, `arm_episode.png` (the greedy end-effector path).
2. **Ablation — no target network** (150 episodes): `--no-target-net` freezes the target net at initialisation and never syncs it.
3. **Ablation — tiny replay** (150 episodes): `--small-buffer` shrinks the buffer to 100 transitions (the warm-up threshold shrinks with it, so training still starts).
4. Optional: repeat the main run with `--episodes 150` to compare against the ablations at equal budget (default @150 gives greedy success 40%).

---

## 5. Self-check (reference numbers)

Produced on CPU (torch 2.15.0+cpu) with the built-in seeds (buffer seed 0, agent seed 0, env seed = episode index; eval seeds 10000–10004).

**FK self-test** — your T1 must reproduce all three ground truths:

```
FK(0, 0)   = (0.80, 0.00)   # arm fully stretched along +x
FK(π/2, 0) = (0.00, 0.80)   # whole arm straight up  (0, L1+L2)
FK(0, π)   = (0.10, 0.00)   # link 2 folded back      (L1−L2, 0)
random policy over 10 episodes: successes=0/10, mean final dist=0.751 m
```

**Main run (300 episodes, CPU)** — the 12 evaluation lines:

```
ep   25/300  eps=0.91  return(20ep)= -67.77  greedy success= 20%
ep   50/300  eps=0.81  return(20ep)= -75.70  greedy success= 40%
ep   75/300  eps=0.71  return(20ep)= -59.38  greedy success= 20%
ep  100/300  eps=0.61  return(20ep)= -52.64  greedy success= 60%
ep  125/300  eps=0.51  return(20ep)= -34.38  greedy success= 80%
ep  150/300  eps=0.41  return(20ep)= -32.94  greedy success= 40%
ep  175/300  eps=0.31  return(20ep)= -25.84  greedy success= 40%
ep  200/300  eps=0.21  return(20ep)= -28.68  greedy success= 60%
ep  225/300  eps=0.11  return(20ep)= -21.40  greedy success=100%
ep  250/300  eps=0.05  return(20ep)= -23.68  greedy success= 80%
ep  275/300  eps=0.05  return(20ep)= -18.11  greedy success= 80%
ep  300/300  eps=0.05  return(20ep)= -19.60  greedy success=100%
```

Headline: **final greedy success = 100%** (5/5 eval episodes), mean return improves −67.8 → −19.6, wall ≈ 78 s on one CPU core. Success-rate trend by quarter of training: **27% → 60% → 67% → 87%**.

Honest note: the post-training showcase episode (`seed=123`, printed at the end) missed at **0.068 m** — just outside the 0.05 m threshold. The greedy policy is good, not infallible; the plot `arm_episode.png` shows the near-miss. Discuss this in Q6.

**Ablations @150 episodes** (default run at 150: 40%):

| run | greedy success per eval (25-ep grid) | final | wall |
|---|---|---|---|
| `--no-target-net` | 20, 20, 0, 20, 20, 20% | **20%** | 43 s |
| `--small-buffer` | 0, 0, 20, 20, 0, 0% | **0%** | 52 s |

---

## 6. Deliverables & Discussion Questions

Record your experimental findings and answer the following questions in your lab notes:

**Deliverables checklist**
- [ ] Completed `reach_env.py` and `dqn.py`.
- [ ] `dqn_learning.png` from the main run, plus one learning-curve plot per ablation.
- [ ] `arm_episode.png` and a one-sentence caption interpreting the path.
- [ ] Console logs of all runs.
- [ ] Answers to Q1–Q7.
- [ ] **Reproducibility block:** commands, seeds (all built-in; state if you changed any), torch version + device, local vs Colab.

**Questions**
- **Q1 — Replay.** Why does DQN need a replay buffer at all (two reasons: sample correlation and data reuse)? Explain your ring-buffer `push`: what does `pos % capacity` buy you over a Python list with `pop(0)`?
- **Q2 — Targets.** Write down your T5 target and explain each factor: the (1−done) mask, the max over a′, and the *target network*. What goes wrong if the network bootstraps from itself every step (non-stationary regression target — "chasing your own tail")?
- **Q3 — Ablations.** `--no-target-net` (frozen at init) still learns (20% vs 40% default): why is a *fixed random* feature map better than nothing, yet worse than periodic syncing? For `--small-buffer`, returns improve (−68 → −42) but success stays ≤20%: relate the buffer size (100 transitions ≈ 1 episode) to overfitting the most recent experience.
- **Q4 — Truncation ≠ termination.** `train` pushes `done=float(info["success"])`, NOT the loop's `done` flag. Explain the difference between the episode *ending because the robot succeeded* and *ending because time ran out*, and what would systematically break if timeouts were treated as terminals (hint: what value would the last 100-step states learn?).
- **Q5 — Loss.** Why `smooth_l1_loss` (Huber) rather than MSE for the TD residual? What kind of TD outliers do you expect early in training?
- **Q6 — Evaluation.** The eval estimates greedy success from 5 episodes (20% granularity) — explain the noisy line in your plot (e.g. 80% → 40%). Then discuss the seed-123 near-miss: is a 0.068 m final distance a policy failure, a threshold artefact, or a seed-lucky/unlucky draw? How would you make the evaluation statistically stronger?
- **Q7 — Hardware.** If you ran both CPU and GPU (Colab T4 or local), record wall-clock for each and explain why a tiny DQN barely benefits from a GPU.

**Scientific rigor:** every claim checkable against a plot or log, seeds/commands stated, deviations recorded honestly.

---

## 7. Going further (optional)

- **Double DQN** (decouple action selection and evaluation in T5) — one line change; does the 225→250 dip smooth out?
- **Dueling networks** or a **prioritised** replay buffer (rank by |TD error|).
- Increase the target radius to 0.75 m (near the arm's max reach 0.8 m) and watch the success ceiling — connect to Week 04, where the same near-limit geometry gets a safety treatment.

## 8. References

- Mnih et al. (2015), *Human-level control through deep reinforcement learning*, Nature 518.
- Sutton & Barto (2nd ed.), §13.3–13.5 (function approximation, DQN); van Hasselt et al. (2016) for Double DQN.
