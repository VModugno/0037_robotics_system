"""Week 06 - Learning from Demonstration, part 4 (advanced): adversarial IRL.

Behaviour cloning treats the expert data as the whole story: if the policy
makes a small mistake it drifts off the expert manifold and the observations
it then sees are unlike anything in the dataset -- *covariate shift*. IRL
instead defines a reward function under which the expert is (approximately)
optimal, letting the policy recover from its own mistakes by *acting* in the
simulator.

This file implements a compact, honest-in-its-simplifications adversarial IRL
loop (AIRL/GAIL family):

    1. Discriminator D(s, a): MLP trained to say "expert" (1) for demo pairs
       and "novice" (0) for policy-rollout pairs                       (T8)
    2. Adversarial reward for the policy: r = -log(1 - D(s, a) + eps)
       -- high where the discriminator is fooled into saying "expert"  (T9)
    3. Generator: the BC MLP (now a Gaussian policy with fixed std) is
       updated with REINFORCE using the adversarial reward             (T10)

Full AIRL adds a value head and disentangles reward from value so the
learned reward transfers across dynamics; we skip that machinery -- the
adversarial core is what matters conceptually. Expect this to be *slower*
and *less stable* than BC on 20 demos: that instability is a finding, not
a bug.

Usage (run AFTER collect/extract/bc, which this script imports):
    pixi run --manifest-path ../pixi.toml -e week06 python airl.py \
        --demos demos/lift_demos_obs.hdf5 --bc-checkpoint bc_policy.pt \
        --iterations 20
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bc import BCNet, build_dataset, obs_features
from collect_demos import make_env

EPS = 1e-8
FIXED_STD = 0.3
GAMMA = 0.99


class Discriminator(nn.Module):
    """MLP on concat(s, a): 19 -> 256 -> 256 -> 1 logit (no sigmoid)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12 + 7, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# TODO T8: discriminator loss
# --------------------------------------------------------------------------- #


def discriminator_loss(logits_expert, logits_novice):
    """TODO T8. Binary-cross-entropy loss of the discriminator.

    Expert pairs are labelled 1, novice (policy rollout) pairs are labelled 0.
    Use F.binary_cross_entropy_with_logits on the two logit batches (targets
    built with torch.ones / torch.zeros) and return the scalar loss.
    """
    raise NotImplementedError("T8: implement the discriminator loss (see docstring)")


# --------------------------------------------------------------------------- #
# TODO T9: adversarial reward
# --------------------------------------------------------------------------- #


def adversarial_reward(logits):
    """TODO T9. Reward for the policy given its discriminator logits.

    r = -log(1 - sigmoid(logits) + 1e-8): near 0 when D says "novice",
    large when the policy manages to fool D into saying "expert".
    Return a tensor with the same shape as `logits`.
    """
    raise NotImplementedError("T9: implement the adversarial reward (see docstring)")


# --------------------------------------------------------------------------- #
# TODO T10: REINFORCE update for the generator
# --------------------------------------------------------------------------- #


def reinforce_update(policy, optimizer, feats, actions, rewards):
    """TODO T10. One REINFORCE update from a batch of rollout transitions.

    Args:
        feats:   (T, 12) observation features for one episode (or a batch).
        actions: (T, 7) actions actually taken in those states.
        rewards: (T,) adversarial rewards, r_t applies to the transition
                 (s_t, a_t).

    Steps:
        1. mean = policy(feats as float32 tensor)
        2. logp = Normal(mean, FIXED_STD).log_prob(actions).sum(dim=-1)
        3. returns: discounted reward-to-go computed *backwards*:
           G[T-1] = r[T-1];  G[t] = r[t] + GAMMA * G[t+1]
        4. loss = -(logp * G).mean(); zero_grad, backward, optimizer step.

    Returns the loss as a float.
    """
    raise NotImplementedError("T10: implement the REINFORCE update (see docstring)")


# --------------------------------------------------------------------------- #
# Rollout collection (complete)
# --------------------------------------------------------------------------- #


def rollout(policy, raw_env, horizon, device, mu, sd):
    """One episode with the Gaussian generator policy; returns (feats, actions)."""
    obs = raw_env.reset()
    feats, actions = [], []
    for _ in range(horizon):
        x = (obs_features(obs) - mu) / (sd + 1e-8)
        with torch.no_grad():
            mean = policy(torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0))
            noise = torch.randn(7, device=device) * FIXED_STD
            a = torch.clamp(mean.squeeze(0) + noise, -1.0, 1.0)
        action = a.cpu().numpy()
        feats.append(x)
        actions.append(action)
        obs, *_ = raw_env.step(action)
        if raw_env._check_success():
            break
    return np.asarray(feats, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Adversarial IRL (AIRL/GAIL-style) on Lift")
    parser.add_argument("--demos", type=str, default="demos/lift_demos_obs.hdf5")
    parser.add_argument("--bc-checkpoint", type=str, default="bc_policy.pt")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--d-steps", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if not os.path.exists(args.demos) or not os.path.exists(args.bc_checkpoint):
        print("[!] need demo file and BC checkpoint (run collect/extract/bc first)")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    X, A = build_dataset(args.demos)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    Xn = ((X - mu) / sd).astype(np.float32)

    # weights_only=False: the checkpoint is our own bc.py output and carries
    # numpy arrays (obs_mean/obs_std) that torch>=2.6 would otherwise reject.
    payload = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
    policy = BCNet().to(device)
    policy.load_state_dict(payload["state_dict"])
    disc = Discriminator().to(device)
    d_opt = torch.optim.Adam(disc.parameters(), lr=1e-4)
    g_opt = torch.optim.Adam(policy.parameters(), lr=1e-4)

    raw_env = make_env(cameras=False, horizon=args.horizon)
    raw_env.reset()

    rng = np.random.default_rng(args.seed)
    print(f"{'iter':>4} {'D_acc':>6} {'nov_r':>7} {'BC-eval':>8}")
    try:
        for it in range(1, args.iterations + 1):
            # -- collect fresh novice rollouts (the generator's data) -------
            rollouts = []
            for _ in range(args.rollouts):
                np.random.seed(int(rng.integers(0, 2**31 - 1)))
                rollouts.append(rollout(policy, raw_env, args.horizon, device, mu, sd))

            # -- discriminator training --------------------------------------
            d_accs = []
            for _ in range(args.d_steps):
                e_idx = rng.integers(0, len(Xn), size=128)
                n_batch = [rollouts[i] for i in rng.integers(0, len(rollouts), size=2)]
                nf = np.concatenate([r[0] for r in n_batch])
                na = np.concatenate([r[1] for r in n_batch])
                n_idx = rng.integers(0, len(nf), size=128)
                se = torch.as_tensor(Xn[e_idx], device=device)
                ae = torch.as_tensor(A[e_idx], dtype=torch.float32, device=device)
                sn = torch.as_tensor(nf[n_idx], device=device)
                an = torch.as_tensor(na[n_idx], device=device)
                d_opt.zero_grad()
                loss = discriminator_loss(disc(se, ae), disc(sn, an))
                loss.backward()
                d_opt.step()
                with torch.no_grad():
                    acc = 0.5 * (
                        (disc(se, ae) > 0).float().mean().item()
                        + (disc(sn, an) < 0).float().mean().item()
                    )
                    d_accs.append(acc)

            # -- generator training (REINFORCE on adversarial reward) -------
            g_losses, nov_rs = [], []
            for feats, actions in rollouts:
                with torch.no_grad():
                    logits = disc(
                        torch.as_tensor(feats, device=device),
                        torch.as_tensor(actions, device=device),
                    )
                    rewards = adversarial_reward(logits)
                nov_rs.append(rewards.mean().item())
                g_opt.zero_grad()
                g_losses.append(
                    reinforce_update(
                        policy,
                        g_opt,
                        torch.as_tensor(feats, device=device),
                        torch.as_tensor(actions, device=device),
                        rewards,
                    )
                )

            # -- closed-loop success check ------------------------------------
            successes = 0
            for _ in range(5):
                np.random.seed(int(rng.integers(0, 2**31 - 1)))
                f, _a = rollout(policy, raw_env, args.horizon, device, mu, sd)
                if raw_env._check_success():
                    successes += 1
            print(
                f"{it:4d} {np.mean(d_accs):6.2%} {np.mean(nov_rs):7.3f} "
                f"{successes}/5      (g_loss {np.mean(g_losses):+.4f})"
            )
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    Complete TODOs T8/T9/T10 in airl.py, then re-run.")
        sys.exit(1)

    torch.save(
        {"state_dict": policy.state_dict(), "obs_mean": mu, "obs_std": sd},
        "airl_policy.pt",
    )
    print("saved airl_policy.pt")


if __name__ == "__main__":
    main()
