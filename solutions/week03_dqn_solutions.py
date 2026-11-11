"""Instructor solutions for COMP0037 Week 03 (week03_dqn).

Reference implementations for every student TODO are monkey-patched onto the
lab modules, smoke-tested, then the lab's own dqn.main() runs (CLI flags pass
through) to reproduce the reference numbers in week03_dqn/LAB.md (self-check):

  T1  reach_env.forward_kinematics   (module-level function)
  T2  ReplayBuffer.push              (ring buffer)
  T3  ReplayBuffer.sample            (uniform -> torch tensor dict)
  T4  DQNAgent.act                   (eps-greedy, single obs, no_grad)
  T5  DQNAgent.td_target             (y = r + gamma*(1-d)*max target_net(s'))
  T6  DQNAgent.optimize              (gather, Huber loss, grad clip 10, Adam)

INSTRUCTOR USE ONLY -- students implement the TODOs themselves.
Artifacts (dqn_final.pt, dqn_learning.png, arm_episode.png) land in the
current working directory.

Run from the labs/ root:
    pixi run --manifest-path pixi.toml -e week03 python solutions/week03_dqn_solutions.py --device cpu --episodes 300
    ... --device cpu --episodes 150 --no-target-net
    ... --device cpu --episodes 150 --small-buffer
"""
from __future__ import annotations

import math
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")    # headless savefig; lab never plt.show()s without --show
sys.dont_write_bytecode = True                # never write __pycache__ into the lab folder

LABS_W03 = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "week03_dqn")
)
sys.path.insert(0, LABS_W03)

import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import torch.nn.functional as F         # noqa: E402

import reach_env                        # noqa: E402
import dqn                              # noqa: E402
from reach_env import L1, L2            # noqa: E402

# ---------------------------------------------------------------- T1
def _fk(theta1: float, theta2: float) -> tuple[float, float]:
    """x = L1 cos(t1) + L2 cos(t1+t2); y = L1 sin(t1) + L2 sin(t1+t2)."""
    return (
        L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2),
        L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2),
    )


reach_env.forward_kinematics = _fk  # _ee / plot_arm / dqn.main's late import all see this


# ---------------------------------------------------------------- T2
def _push(self: dqn.ReplayBuffer, obs: np.ndarray, action: int, reward: float,
          next_obs: np.ndarray, done: float) -> None:
    """Ring-buffer insert: write at pos, advance circularly, cap size."""
    self.obs[self.pos] = obs
    self.actions[self.pos] = action
    self.rewards[self.pos] = reward
    self.next_obs[self.pos] = next_obs
    self.dones[self.pos] = done
    self.pos = (self.pos + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)


dqn.ReplayBuffer.push = _push


# ---------------------------------------------------------------- T3
def _sample(self: dqn.ReplayBuffer, batch_size: int) -> dict[str, torch.Tensor]:
    """Uniform minibatch as CPU torch tensors (obs (B,6) f32, actions (B,1) i64,
    rewards (B,) f32, next_obs (B,6) f32, dones (B,) f32)."""
    idx = self.rng.integers(0, self.size, size=batch_size)
    return {
        "obs": torch.from_numpy(self.obs[idx]),
        "actions": torch.from_numpy(np.expand_dims(self.actions[idx], 1)),
        "rewards": torch.from_numpy(self.rewards[idx]),
        "next_obs": torch.from_numpy(self.next_obs[idx]),
        "dones": torch.from_numpy(self.dones[idx]),
    }


dqn.ReplayBuffer.sample = _sample


# ---------------------------------------------------------------- T4
def _act(self: dqn.DQNAgent, obs: np.ndarray, epsilon: float = 0.0) -> int:
    """Epsilon-greedy for a single observation; plain Python int out."""
    if self.rng.random() < epsilon:
        return int(self.rng.integers(self.n_actions))
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        q = self.q(x.unsqueeze(0))
        return int(q.argmax(dim=1).item())


dqn.DQNAgent.act = _act


# ---------------------------------------------------------------- T5
def _td_target(self: dqn.DQNAgent, rewards: torch.Tensor, next_obs: torch.Tensor,
               dones: torch.Tensor) -> torch.Tensor:
    """y = r + gamma * (1 - done) * max_a' q_target(s')[a']  (no gradients)."""
    with torch.no_grad():
        q_next = self.q_target(next_obs).max(dim=1).values
        return rewards + self.gamma * (1.0 - dones) * q_next


dqn.DQNAgent.td_target = _td_target


# ---------------------------------------------------------------- T6
def _optimize(self: dqn.DQNAgent, batch: dict[str, torch.Tensor]) -> float:
    """One gradient step on the TD (Huber) loss."""
    obs = batch["obs"].to(self.device)
    actions = batch["actions"].to(self.device)
    rewards = batch["rewards"].to(self.device)
    next_obs = batch["next_obs"].to(self.device)
    dones = batch["dones"].to(self.device)

    q_selected = self.q(obs).gather(1, actions).squeeze(1)   # (B,)
    y = self.td_target(rewards, next_obs, dones)             # (B,), no grad
    loss = F.smooth_l1_loss(q_selected, y)

    self.opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
    self.opt.step()
    return float(loss.item())


dqn.DQNAgent.optimize = _optimize


# ---------------------------------------------------------------- pre-flight
def smoke_test() -> None:
    """Exercise every patched TODO on synthetic data before the long run."""
    # T1
    x, y = reach_env.forward_kinematics(0.0, 0.0)
    assert abs(x - (L1 + L2)) < 1e-9 and abs(y) < 1e-9, "FK(0,0) wrong"
    x, y = reach_env.forward_kinematics(0.0, math.pi)
    assert abs(x - (L1 - L2)) < 1e-9 and abs(y) < 1e-9, "FK(0,pi) wrong"

    # T2 ring behaviour: overfill a capacity-8 buffer with 10 pushes
    buf = dqn.ReplayBuffer(capacity=8)
    for i in range(10):
        buf.push(np.full(6, float(i), dtype=np.float32), i % 9, float(i),
                 np.full(6, float(i) + 1.0, dtype=np.float32), float(i % 3 == 0))
    assert len(buf) == 8, f"ring size wrong: {len(buf)}"
    assert buf.obs[2][0] == 2.0, "oldest surviving entry should be push i=2"
    assert buf.obs[1][0] == 9.0, "newest entry (i=9) should sit at index 1"
    assert buf.pos == 2 and buf.size == 8, "pos/size after wrap wrong"

    # T3 shapes/dtypes
    b = buf.sample(4)
    assert b["obs"].shape == (4, 6) and b["obs"].dtype == torch.float32
    assert b["actions"].shape == (4, 1) and b["actions"].dtype == torch.int64
    assert b["rewards"].shape == (4,) and b["rewards"].dtype == torch.float32
    assert b["next_obs"].shape == (4, 6) and b["next_obs"].dtype == torch.float32
    assert b["dones"].shape == (4,) and b["dones"].dtype == torch.float32

    # T4
    agent = dqn.DQNAgent(device="cpu")
    a = agent.act(np.zeros(6, dtype=np.float32), epsilon=1.0)
    assert isinstance(a, int) and 0 <= a < 9
    a = agent.act(np.zeros(6, dtype=np.float32), epsilon=0.0)
    assert isinstance(a, int) and 0 <= a < 9

    # T5
    y = agent.td_target(b["rewards"], b["next_obs"], b["dones"])
    assert y.shape == (4,) and not y.requires_grad

    # T6
    loss = agent.optimize(buf.sample(4))
    assert isinstance(loss, float)

    print("smoke test: T1-T6 reference solutions OK "
          f"(one optimize step on a random batch, loss={loss:.4f})")


def main() -> None:
    t0 = time.time()
    print(f"week03_dqn_solutions.py | cwd: {os.getcwd()}")
    print(f"torch {torch.__version__} | numpy {np.__version__}")
    smoke_test()
    print("-" * 64)
    # dqn.main() argparse reads sys.argv; pass our CLI flags straight through
    sys.argv = ["dqn.py"] + sys.argv[1:]
    dqn.main()
    print(f"total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
