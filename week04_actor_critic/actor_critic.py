"""
COMP0037 — Week 04 Lab
=======================
Actor-Critic (A2C) with basic action constraints, plus a CBF-style safety
shield, on the continuous 2-link arm.

YOUR TASKS (see LAB.md):
  * T1 — `GaussianActor.sample_action`   (stochastic policy)
  * T2 — `compute_gae`                   (advantage estimation, the heart of A2C)
  * T3 — `cbf_safe_action`               (Control-Barrier-Function-style shield)
  * T4 — the policy-gradient loss line in `A2CAgent.update`

Everything else (critic, rollout, training loop, evaluation, plots) is wired.

Run me:
    pixi run -e week04 python actor_critic.py            # trains safety ON vs OFF
    pixi run -e week04 python actor_critic.py --episodes 500

Produces: a2c_comparison.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from arm2d_env import DT, JOINT_LIMITS, OMEGA_MAX, Arm2DEnv


# --------------------------------------------------------------------- actor
class GaussianActor(nn.Module):
    """pi(a|s) = Normal(mean=MLP(s), std=exp(log_std)) — a diagonal Gaussian."""

    def __init__(self, obs_dim: int = 6, action_dim: int = 2, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.tensor(-0.5))  # state-independent std

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = self.log_std.exp().clamp(1e-3, 10.0).expand_as(mean)
        return Normal(mean, std)

    def sample_action(self, obs: np.ndarray) -> np.ndarray:
        """Draw one action for one observation (TODO T1).

        1. obs (numpy (6,)) -> torch float32 tensor with a batch dimension
        2. dist = self.distribution(x)
        3. a = dist.rsample()   (reparameterised: gradients could flow through it)
        4. return a[0].cpu().numpy()  as a (2,) numpy array

        (The rollout calls this under torch.no_grad(); log-probabilities are
        recomputed later on the whole episode — that is why rsample + storing
        the action is enough.)
        """
        raise NotImplementedError(
            "TODO(T1): implement GaussianActor.sample_action — see LAB.md."
        )

    def mean_action(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action for evaluation (COMPLETE)."""
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.net(x)[0].cpu().numpy()


class Critic(nn.Module):
    """V(s): state -> one scalar value (COMPLETE)."""

    def __init__(self, obs_dim: int = 6, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------- advantages
def compute_gae(
    rewards: np.ndarray,      # (T,)
    values: np.ndarray,       # (T,)   V(s_t)
    next_values: np.ndarray,  # (T,)   V(s_{t+1}), or 0 where the step was terminal
    dones: np.ndarray,        # (T,)   1.0 only on true terminal steps
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation, backwards over one episode (TODO T2).

        delta_t = r_t + gamma * next_values[t] - values[t]     # TD residual
        A_t     = delta_t + gamma * lam * A_{t+1}              # accumulate backwards
        returns = A + values                                    # bootstrapped targets

    Start from A_{T} = 0 and iterate t = T-1 ... 0.
    Return (advantages, returns), both (T,) float arrays.
    """
    raise NotImplementedError(
        "TODO(T2): implement compute_gae — see LAB.md."
    )


# ------------------------------------------------------------ safety shield
def cbf_safe_action(
    theta: np.ndarray,        # (2,) current joint angles
    omega: np.ndarray,        # (2,) commanded joint velocities (rad/s)
    limits: tuple[float, float] = JOINT_LIMITS,
    h_min: float = 0.05,      # safety margin (rad): stop this short of |limit|
    dt: float = DT,
) -> tuple[np.ndarray, int]:
    """CBF-style action shield for the joint limits (TODO T3).

    The barrier condition we want to enforce for each joint i:

        h_i(x) = limit_i - |theta_i|  >=  h_min     (stay inside the safe set)

    The unshielded controller may violate it; we *project* the commanded
    velocity onto the safe set:

      for each joint i:
        margin_i = limit_i - h_min            # safe boundary for |theta_i|
        overshoot = |theta_i| + |omega_i| * dt - margin_i
        if overshoot > 0 and omega_i pushes |theta_i| outward:
            clip |omega_i| to (margin_i - |theta_i|) / dt   (may be negative -> 0)

    Return (omega_safe (2,), n_clipped) — the number of joints you clipped.
    """
    raise NotImplementedError(
        "TODO(T3): implement cbf_safe_action — see LAB.md."
    )


def safety_shield(theta: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, int]:
    """Wrap the shield around actions in [-1, 1]^2 units (COMPLETE)."""
    omega = np.asarray(action, dtype=np.float64) * OMEGA_MAX
    omega_safe, n = cbf_safe_action(theta, omega)
    return omega_safe / OMEGA_MAX, n


# -------------------------------------------------------------------- agent
class A2CAgent:
    def __init__(self, device: str = "cpu", actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3, gamma: float = 0.99,
                 lam: float = 0.95, entropy_coef: float = 0.01) -> None:
        self.device = torch.device(device)
        self.actor = GaussianActor().to(self.device)
        self.critic = Critic().to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma, self.lam, self.entropy_coef = gamma, lam, entropy_coef

    def rollout(self, env: Arm2DEnv, seed: int, safety: bool) -> dict:
        """Collect one episode of experience (COMPLETE — read it)."""
        obs_list, act_list, rew_list, done_list = [], [], [], []
        obs, done = env.reset(seed=seed), False
        ep_ret, shielded = 0.0, 0
        final_obs = None
        while not done:
            with torch.no_grad():
                a = self.actor.sample_action(obs)          # (2,), unbounded
            a = np.clip(a, -1.0, 1.0)                      # basic action constraint
            if safety:
                a, n = safety_shield(env.thetas, a)         # CBF-style shield
                shielded += n
            nobs, r, done, info = env.step(a)
            obs_list.append(obs)
            act_list.append(a)
            rew_list.append(r)
            # 'done' for learning is TRUE only on success (termination);
            # a time limit is a truncation — the episode's return just stops,
            # the MDP itself continues (same subtlety as Week 03, report Q4).
            done_list.append(float(info["success"]))
            ep_ret += r
            obs = nobs
            final_obs = nobs
        return {
            "obs": np.asarray(obs_list, dtype=np.float32),
            "actions": np.asarray(act_list, dtype=np.float32),
            "rewards": np.asarray(rew_list, dtype=np.float64),
            "dones": np.asarray(done_list, dtype=np.float64),
            "final_obs": final_obs,
            "return": ep_ret,
            "success": bool(info["success"]),
            "limit_hits": env.limit_hits,
            "shielded": shielded,
        }

    def update(self, tr: dict) -> tuple[float, float]:
        """One A2C update from a single episode (T is the episode length)."""
        obs_t = torch.as_tensor(tr["obs"], device=self.device)
        act_t = torch.as_tensor(tr["actions"], device=self.device)
        rew_np = tr["rewards"]
        done_np = tr["dones"]

        values = self.critic(obs_t)                       # (T,), with grad
        with torch.no_grad():
            values_np = values.cpu().numpy()
            if bool(done_np[-1]):
                last_v = 0.0
            else:                                          # truncated: bootstrap
                f = torch.as_tensor(tr["final_obs"], dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
                last_v = float(self.critic(f).item())
            next_values_np = np.append(values_np[1:], last_v)

        adv_np, ret_np = compute_gae(rew_np, values_np, next_values_np, done_np,
                                     self.gamma, self.lam)
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)   # normalise
        adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret_np, dtype=torch.float32, device=self.device)

        dist = self.distribution_of(obs_t)
        logp = dist.log_prob(act_t).sum(-1)               # (T,)
        entropy = dist.entropy().sum(-1).mean()           # scalar

        # ── TODO(T4): the policy-gradient loss, one line ────────────────────
        # actor_loss = -(logp * adv_t).mean() - self.entropy_coef * entropy
        # (surrogate objective: push up the log-probability of actions that
        #  did better than expected; keep some entropy for exploration)
        raise NotImplementedError(
            "TODO(T4): implement the actor loss in A2CAgent.update — see LAB.md."
        )
        # ─────────────────────────────────────────────────────────────────────

        critic_loss = nn.functional.mse_loss(values, ret_t)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        return float(actor_loss.item()), float(critic_loss.item())

    def distribution_of(self, obs_t: torch.Tensor) -> Normal:
        return self.actor.distribution(obs_t)


# ------------------------------------------------------------------ training
def evaluate(env: Arm2DEnv, agent: A2CAgent, episodes: int = 10,
             safety: bool = False) -> float:
    """Greedy (mean-action) success rate (COMPLETE)."""
    wins = 0
    for ep in range(episodes):
        obs, done = env.reset(seed=20_000 + ep), False
        while not done:
            a = np.clip(agent.actor.mean_action(obs), -1.0, 1.0)
            if safety:
                a, _ = safety_shield(env.thetas, a)
            obs, _, done, info = env.step(a)
        wins += bool(info["success"])
    return wins / episodes


def train(env: Arm2DEnv, agent: A2CAgent, episodes: int = 400,
          safety: bool = False, eval_every: int = 25) -> dict:
    """Train one A2C agent; safety toggles the CBF shield (COMPLETE)."""
    returns, hits, succ_hist = [], [], []
    t0 = time.time()
    for ep in range(episodes):
        tr = agent.rollout(env, seed=ep, safety=safety)
        try:
            agent.update(tr)
        except NotImplementedError as e:
            raise SystemExit(f"\n[!] {e}\n    Complete TODOs T1-T4 first.") from e
        returns.append(tr["return"])
        hits.append(tr["limit_hits"])
        if (ep + 1) % eval_every == 0:
            sr = evaluate(env, agent, safety=safety)
            succ_hist.append((ep, sr))
            print(f"[{'shield' if safety else 'plain '} ep {ep + 1:>4}/{episodes}]  "
                  f"return(20)={np.mean(returns[-20:]):6.2f}  "
                  f"limit hits/ep={np.mean(hits[-20:]):5.1f}  "
                  f"greedy success={sr:4.0%}  [{time.time() - t0:5.0f}s]")
    return {"returns": returns, "hits": hits, "success": succ_hist}


def main() -> None:
    ap = argparse.ArgumentParser(description="Week 04 Actor-Critic lab")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   (torch {torch.__version__})")
    torch.manual_seed(0)  # LAB.md reference numbers were produced with this seed
    np.random.seed(0)

    # quick check the TODOs are done
    probe_env, probe_agent = Arm2DEnv(), A2CAgent(device=device)
    try:
        tr = probe_agent.rollout(probe_env, seed=0, safety=True)
        probe_agent.update(tr)
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        return

    configs = {"no shield (unsafe)": False, "CBF shield (safe)": True}
    results = {}
    for label, safety in configs.items():
        print(f"\n-- training with {label} {'-' * 44}")
        env, agent = Arm2DEnv(), A2CAgent(device=device)
        results[label] = train(env, agent, episodes=args.episodes, safety=safety)
        results[label]["agent"] = agent

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for label, hist in results.items():
        ma = np.convolve(hist["returns"], np.ones(25) / 25, mode="valid")
        axes[0].plot(range(24, len(hist["returns"])), ma, label=label)
        axes[1].plot(np.cumsum(hist["hits"]), label=label)
        if hist["success"]:
            eps_, srs = zip(*hist["success"])
            axes[2].plot(eps_, np.array(srs) * 100, "o-", label=label)
    axes[0].set_title("training return (25-ep moving avg)")
    axes[0].set_xlabel("episode")
    axes[1].set_title("cumulative joint-limit violations")
    axes[1].set_xlabel("episode")
    axes[2].set_title("greedy success rate (%)")
    axes[2].set_xlabel("episode")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("a2c_comparison.png", dpi=150)
    print("\nsaved: a2c_comparison.png")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
