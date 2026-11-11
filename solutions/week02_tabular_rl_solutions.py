"""
Instructor solutions for COMP0037 Week 02 (week02_tabular_rl).

Reference implementations for TODOs T1-T3 are monkey-patched onto the agents
module exactly as specified in its docstrings, then run_lab.main() runs with
default parameters to reproduce the reference numbers in
week02_tabular_rl/LAB.md (self-check).

INSTRUCTOR USE ONLY -- students implement the TODOs themselves.

Import-style note (verified against run_lab.py):
    run_lab.py does `from agents import train`. That binds only the `train`
    function object; `train` is defined in agents.py, so its __globals__ IS
    agents.__dict__, and it resolves select_epsilon_greedy /
    q_learning_update / sarsa_update through that module dict at CALL time.
    Patching agents.<attr> therefore reaches the code that matters.

Run from the labs/ root:
    pixi run --manifest-path pixi.toml -e week02 python solutions/week02_tabular_rl_solutions.py
"""

import os
import sys
import time

LABS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "week02_tabular_rl")
)
sys.path.insert(0, LABS)

import numpy as np

import gridworld  # noqa: F401  (complete file, no TODOs)
import agents     # the module carrying TODOs T1-T3


# ---- T1: epsilon-greedy behaviour policy ----------------------------------
def select_epsilon_greedy(Q, s_idx, epsilon, rng):
    """With probability `epsilon`: a uniformly random action; else greedy argmax."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[s_idx]))


# ---- T2: Q-Learning update (off-policy TD control) -------------------------
def q_learning_update(Q, s_idx, a, r, s2_idx, done, alpha=0.5, gamma=1.0):
    """target = r + gamma * max_a' Q[s2_idx, a']  (target = r if done)."""
    target = r if done else r + gamma * float(np.max(Q[s2_idx]))
    Q[s_idx, a] += alpha * (target - Q[s_idx, a])


# ---- T3: Sarsa update (on-policy TD control) --------------------------------
def sarsa_update(Q, s_idx, a, r, s2_idx, a2, done, alpha=0.5, gamma=1.0):
    """target = r + gamma * Q[s2_idx, a2]  (target = r if done)."""
    target = r if done else r + gamma * float(Q[s2_idx, a2])
    Q[s_idx, a] += alpha * (target - Q[s_idx, a])


# Monkey-patch the solved implementations onto the agents module (exact
# signatures from agents.py). run_lab's `train` resolves these names through
# agents.__dict__ at call time, so these are the versions actually used.
agents.select_epsilon_greedy = select_epsilon_greedy
agents.q_learning_update = q_learning_update
agents.sarsa_update = sarsa_update

# NOW import the driver and run it with default parameters.
import run_lab  # noqa: E402

if __name__ == "__main__":
    t0 = time.perf_counter()
    run_lab.main()
    print(f"\n[verify] solved run wall time: {time.perf_counter() - t0:.1f}s")
