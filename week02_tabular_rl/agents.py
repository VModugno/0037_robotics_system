"""
COMP0037 — Week 02 Lab
=======================
YOUR implementations: epsilon-greedy, Q-Learning, and Sarsa.

Everything else (environments, training loop, experiments) is provided in
gridworld.py and run_lab.py — this file is where you do the thinking.

YOUR TASKS (see LAB.md):
  * T1 — `select_epsilon_greedy` : the behaviour policy
  * T2 — `q_learning_update`     : off-policy TD control
  * T3 — `sarsa_update`          : on-policy TD control
"""

from __future__ import annotations

import numpy as np

# Q has shape (n_states, n_actions) — the action-value table Q[s, a].


def select_epsilon_greedy(
    Q: np.ndarray, s_idx: int, epsilon: float, rng: np.random.Generator
) -> int:
    """Behaviour policy shared by Q-Learning and Sarsa (TODO T1).

    * with probability `epsilon`: a uniformly random action
    * otherwise: the greedy action  argmax_a Q[s_idx, a]

    Return the action index (an int).
    """
    raise NotImplementedError(
        "TODO(T1): implement select_epsilon_greedy — see LAB.md."
    )


def q_learning_update(
    Q: np.ndarray,
    s_idx: int,
    a: int,
    r: float,
    s2_idx: int,
    done: bool,
    alpha: float = 0.5,
    gamma: float = 1.0,
) -> None:
    """One step of Q-Learning (off-policy TD control) — update Q in place (TODO T2).

    The learning target is the *greedy* bootstrap, regardless of what the
    behaviour policy does next:

        target = r + gamma * max_a' Q[s2_idx, a']        (if not done)
        target = r                                        (if done)

        Q[s_idx, a] += alpha * (target - Q[s_idx, a])

    Note: `done` refers to the transition having reached a terminal state —
    when done is True there is no next state to bootstrap from.
    """
    raise NotImplementedError(
        "TODO(T2): implement q_learning_update — see LAB.md."
    )


def sarsa_update(
    Q: np.ndarray,
    s_idx: int,
    a: int,
    r: float,
    s2_idx: int,
    a2: int,
    done: bool,
    alpha: float = 0.5,
    gamma: float = 1.0,
) -> None:
    """One step of Sarsa (on-policy TD control) — update Q in place (TODO T3).

    The learning target uses the action the behaviour policy ACTUALLY takes
    next (a2), not the greedy one:

        target = r + gamma * Q[s2_idx, a2]               (if not done)
        target = r                                        (if done)

        Q[s_idx, a] += alpha * (target - Q[s_idx, a])
    """
    raise NotImplementedError(
        "TODO(T3): implement sarsa_update — see LAB.md."
    )


# --------------------------------------------------------------- training
def train(
    env,
    method: str = "q_learning",
    episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Training loop for both algorithms (COMPLETE — read it!).

    The loop is deliberately the *same* for Q-Learning and Sarsa; the two
    algorithms differ only in the update rule they call (your T2 vs T3).
    Both use the same epsilon-greedy behaviour policy (your T1).

    This is the canonical single-step TD control loop:

        a  = behaviour(s)
        loop: s2, r, done = step(s, a)
              a2 = behaviour(s2)              # what we'd ACTUALLY do next
              method == "sarsa"     -> sarsa_update(..., a2, ...)
              method == "q_learning" -> q_learning_update(..., max over a' inside)
              s, a = s2, a2

    Returns (Q, per-episode undiscounted returns).
    """
    assert method in ("q_learning", "sarsa")
    env.reset(seed)
    rng = np.random.default_rng(seed + 7)
    Q = np.zeros((env.n_states, env.n_actions))
    returns: list[float] = []

    for _ in range(episodes):
        s = env.reset()
        ep_return = 0.0
        done = False
        a = select_epsilon_greedy(Q, s, epsilon, rng)
        while not done:
            s2, r, done = env.step(s, a)
            ep_return += r
            a2 = select_epsilon_greedy(Q, s2, epsilon, rng) if not done else -1
            if method == "q_learning":
                q_learning_update(Q, s, a, r, s2, done, alpha, gamma)
            else:
                sarsa_update(Q, s, a, r, s2, a2, done, alpha, gamma)
            s, a = s2, a2
        returns.append(ep_return)

    return Q, returns
