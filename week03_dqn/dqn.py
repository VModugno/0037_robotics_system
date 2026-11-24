"""
COMP0037 — Week 03 Lab
=======================
Building a PyTorch DQN agent for the discrete robotic reaching task.

YOUR TASKS (see LAB.md):
  * T2 — `ReplayBuffer.push`       (ring buffer)
  * T3 — `ReplayBuffer.sample`     (uniform sampling -> torch tensors)
  * T4 — `DQNAgent.act`            (epsilon-greedy action selection)
  * T5 — `DQNAgent.td_target`      (the bootstrapped regression target)
  * T6 — `DQNAgent.optimize`       (one gradient step of the TD loss)

The training loop, evaluation, plots, and the two stabilising tricks
(experience replay + target network) are wired up for you — you implement
their inner mechanics.

Run me:
    pixi run -e week03 python dqn.py --episodes 300
    pixi run -e week03 python dqn.py --no-target-net   # ablation (discussion Q3)
    pixi run -e week03 python dqn.py --small-buffer    # ablation (discussion Q3)

Produces: dqn_learning.png, dqn_final.pt, arm_episode.png
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reach_env import ACTION_DELTAS, ArmReachEnv


# ------------------------------------------------------------------ network
class QNet(nn.Module):
    """A small MLP: state (6,) -> Q-value for each of the 9 actions (COMPLETE)."""

    def __init__(self, obs_dim: int = 6, n_actions: int = 9, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------------------------------------- replay buffer
class ReplayBuffer:
    """A fixed-size ring buffer of transitions (TODO T2, T3).

    Stored in flat numpy arrays so `sample` is cheap. One entry:
        (obs[6], action, reward, next_obs[6], done)
    `done` is 1.0 ONLY for true terminal transitions (success) — time-limit
    truncations must bootstrap (see train() and discussion Q4!).
    """

    def __init__(self, capacity: int = 10_000, obs_dim: int = 6, seed: int = 0) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: bool) -> None:
        """Store one transition, overwriting the oldest when full (TODO T2).

        Write at index self.pos, then advance self.pos circularly
        (modulo self.capacity) and keep self.size = min(size+1, capacity).
        """
        raise NotImplementedError("TODO(T2): implement ReplayBuffer.push — see LAB.md.")

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a random minibatch and return it as CPU torch tensors (TODO T3).

        Draw `batch_size` indices uniformly from [0, self.size), gather the
        rows, and return a dict:
            obs      (B, 6)  float32
            actions  (B, 1)  int64     <- shape (B, 1), for torch.gather
            rewards  (B,)    float32
            next_obs (B, 6)  float32
            dones    (B,)    float32
        Hint: np.expand_dims(self.actions[idx], 1) gives the (B, 1) shape.
        """
        raise NotImplementedError("TODO(T3): implement ReplayBuffer.sample — see LAB.md.")


# --------------------------------------------------------------------- agent
class DQNAgent:
    """DQN with experience replay + a target network (TODO T4-T6)."""

    def __init__(self, obs_dim: int = 6, n_actions: int = 9,
                 device: str = "cpu", lr: float = 1e-3, gamma: float = 0.99) -> None:
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.q = QNet(obs_dim, n_actions).to(self.device)
        self.q_target = QNet(obs_dim, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)  # the target net is never trained directly
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.rng = np.random.default_rng(0)

    def sync_target(self) -> None:
        """Copy the online network into the target network (COMPLETE)."""
        self.q_target.load_state_dict(self.q.state_dict())

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action for ONE observation (TODO T4).

        * with probability `epsilon`: a uniformly random action index
        * else: argmax_a q(obs)[a]

        Use torch.no_grad() — action selection never needs gradients.
        Return a plain Python int.
        """
        raise NotImplementedError("TODO(T4): implement DQNAgent.act — see LAB.md.")

    def td_target(self, rewards: torch.Tensor, next_obs: torch.Tensor,
                  dones: torch.Tensor) -> torch.Tensor:
        """The bootstrapped regression target y (TODO T5).

            y = r + gamma * (1 - done) * max_a' q_target(s')[a']

        All inputs are (B,) / (B, 6) tensors already on self.device.
        Compute inside torch.no_grad() — the target must NOT carry gradients.
        Return a (B,) tensor.
        """
        raise NotImplementedError("TODO(T5): implement DQNAgent.td_target — see LAB.md.")

    def optimize(self, batch: dict[str, torch.Tensor]) -> float:
        """One gradient step on a sampled minibatch (TODO T6).

        1. Move every tensor in `batch` to self.device.
        2. q_selected = q(obs).gather(1, actions).squeeze(1)    # (B,)
        3. y = self.td_target(rewards, next_obs, dones)
        4. loss = F.smooth_l1_loss(q_selected, y)               # Huber loss
        5. self.opt.zero_grad(); loss.backward();
           torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0);
           self.opt.step()
        Return the Python float loss.
        """
        raise NotImplementedError("TODO(T6): implement DQNAgent.optimize — see LAB.md.")


# ------------------------------------------------------------------ training
def greedy_success_rate(env: ArmReachEnv, agent: DQNAgent, episodes: int = 5) -> float:
    """Fraction of episodes the *greedy* policy finishes successfully (COMPLETE)."""
    wins = 0
    for ep in range(episodes):
        obs, done = env.reset(seed=10_000 + ep), False
        while not done:
            obs, _, done, info = env.step(agent.act(obs, epsilon=0.0))
        wins += bool(info["success"])
    return wins / episodes


def train(env: ArmReachEnv, agent: DQNAgent, buffer: ReplayBuffer, *,
          episodes: int = 300, batch: int = 64, warmup: int = 200,
          eps_start: float = 1.0, eps_end: float = 0.05, eps_frac: float = 0.8,
          sync_every: int = 10, use_target_net: bool = True,
          log_every: int = 20, eval_every: int = 25) -> dict:
    """The DQN training loop (COMPLETE — but read every line)."""
    returns: list[float] = []
    success_at: list[tuple[int, float]] = []
    t0 = time.time()

    for ep in range(episodes):
        # linear epsilon decay over the first `eps_frac` of training
        frac = min(1.0, ep / (eps_frac * episodes))
        eps = eps_start + frac * (eps_end - eps_start)

        obs, done, ep_ret = env.reset(seed=ep), False, 0.0
        while not done:
            a = agent.act(obs, eps)
            next_obs, r, done, info = env.step(a)
            # truncation is NOT termination: only 'success' ends the MDP episode
            buffer.push(obs, a, r, next_obs, float(info["success"]))
            ep_ret += r
            if len(buffer) >= warmup:
                agent.optimize(buffer.sample(batch))
            obs = next_obs

        returns.append(ep_ret)
        if use_target_net and (ep + 1) % sync_every == 0:
            agent.sync_target()
        if (ep + 1) % eval_every == 0:
            sr = greedy_success_rate(env, agent)
            success_at.append((ep, sr))
            print(f"ep {ep + 1:>4}/{episodes}  "
                  f"eps={eps:.2f}  return(20ep)={np.mean(returns[-20:]):7.2f}  "
                  f"greedy success={sr:4.0%}  [{time.time() - t0:5.0f}s]")

    return {"returns": returns, "success": success_at}


def main() -> None:
    ap = argparse.ArgumentParser(description="Week 03 DQN lab")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-target-net", action="store_true",
                    help="ablation: freeze the target net at init, never sync it (discussion Q3)")
    ap.add_argument("--small-buffer", action="store_true",
                    help="ablation: replay buffer of only 100 (discussion Q3)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   (torch {torch.__version__})")

    env = ArmReachEnv()
    agent = DQNAgent(device=device)
    buffer = ReplayBuffer(capacity=100 if args.small_buffer else 10_000)
    warmup = min(200, buffer.capacity)

    try:
        agent.act(env.reset(), 0.5)  # quick check that T4 exists
    except NotImplementedError as e:
        print(f"\n[!] {e}\n    Complete TODOs T2-T6 in dqn.py first.")
        return

    hist = train(env, agent, buffer, episodes=args.episodes, warmup=warmup,
                 use_target_net=not args.no_target_net)

    torch.save(agent.q.state_dict(), "dqn_final.pt")
    print("saved: dqn_final.pt")

    # ── plot learning curves ───────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(hist["returns"], alpha=0.3, label="per-episode return")
    if len(hist["returns"]) >= 25:
        ma = np.convolve(hist["returns"], np.ones(25) / 25, mode="valid")
        ax1.plot(range(24, len(hist["returns"])), ma, lw=2,
                 label="25-episode moving avg")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("return")
    ax1.set_title("DQN on ArmReachEnv"
                  + (" (no target net!)" if args.no_target_net else "")
                  + (" (buffer=100!)" if args.small_buffer else ""))
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    if hist["success"]:
        eps, srs = zip(*hist["success"])
        ax2.plot(eps, np.array(srs) * 100, "o-")
        ax2.set_ylim(0, 105)
        ax2.set_xlabel("episode")
        ax2.set_ylabel("greedy success rate (%)")
        ax2.set_title("greedy evaluation (5 episodes)")
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("dqn_learning.png", dpi=150)
    print("saved: dqn_learning.png")
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    # ── visualise one greedy episode ────────────────────────────────────
    from reach_env import forward_kinematics

    obs, done = env.reset(seed=123), False
    thetas, ee_path = [], []
    while not done:
        obs, _, done, info = env.step(agent.act(obs, 0.0))
        thetas.append(env.thetas)
        ee_path.append(forward_kinematics(*env.thetas))
    print(f"greedy episode: success={info['success']}  final dist={info['dist']:.3f} m")
    ArmReachEnv.plot_arm(np.asarray(thetas)[-1], env.target, path=ee_path,
                         fname="arm_episode.png")


if __name__ == "__main__":
    main()
