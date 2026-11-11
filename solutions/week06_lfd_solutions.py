"""
Instructor solutions for COMP0037 Week 06 (week06_lfd).

Reference implementations for the ten student TODOs are monkey-patched onto
the lab modules, then the lab mains run stage by stage:

  T1  collect_demos.expert_action           -> module-attribute assignment
  T2  collect_demos.ExpertController.next_action -> method assignment
  T3  extract_obs.replay_observation        -> module-attribute assignment
  T4  bc.build_dataset                      -> module-attribute assignment
  T5  bc.BCNet.forward                      -> method assignment
  T6  bc.train_step                         -> module-attribute assignment
  T7  bc.evaluate_policy                    -> module-attribute assignment
  T8  airl.discriminator_loss               -> module-attribute assignment
  T9  airl.adversarial_reward               -> module-attribute assignment
  T10 airl.reinforce_update                 -> module-attribute assignment

INSTRUCTOR USE ONLY -- students implement the TODOs themselves.

Run from the week06_lfd folder (outputs land next to the student's):
    cd week06_lfd
    pixi run --manifest-path ../pixi.toml -e week06 python ../solutions/week06_lfd_solutions.py --stage collect
    pixi run --manifest-path ../pixi.toml -e week06 python ../solutions/week06_lfd_solutions.py --stage extract
    pixi run --manifest-path ../pixi.toml -e week06 python ../solutions/week06_lfd_solutions.py --stage bc
    pixi run --manifest-path ../pixi.toml -e week06 python ../solutions/week06_lfd_solutions.py --stage airl

--stage all runs collect -> extract -> bc -> airl sequentially.
Reference numbers from a 20-episode camera-less run on the instructor laptop
are reported in week06_lfd/LAB.md (self-check).
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

LAB_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "week06_lfd")
)
sys.path.insert(0, LAB_DIR)

import collect_demos
import extract_obs
import bc
import airl


# --------------------------------------------------------------------------- #
# reference implementations
# --------------------------------------------------------------------------- #


def expert_action(eef_pos, target_pos, grip):
    a = np.zeros(7, dtype=np.float64)
    a[:3] = np.clip((np.asarray(target_pos) - np.asarray(eef_pos)) / 0.05, -1.0, 1.0)
    a[6] = grip
    return a


def _next_action(self, obs):
    self.cube_pos = np.asarray(obs["object-state"], dtype=np.float64)[:3]
    eef = np.asarray(obs["robot0_eef_pos"])
    if self.phase_idx == 2:
        if self.grip_count >= self.GRIP_STEPS:
            self.phase_idx = min(self.phase_idx + 1, len(collect_demos.PHASES) - 1)
        else:
            self.grip_count += 1
    else:
        if np.linalg.norm(self._waypoint() - eef) < self.TOL:
            self.phase_idx = min(self.phase_idx + 1, len(collect_demos.PHASES) - 1)
    return collect_demos.expert_action(eef, self._waypoint(), self._grip())


def replay_observation(env, state):
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    return env._get_observations(force_update=True)


def build_dataset(hdf5_path):
    X_list, A_list = [], []
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        for name in demos:
            grp = f[f"data/{name}"]
            obs_grp = grp["obs"]
            T = len(obs_grp["robot0_eef_pos"])
            for t in range(T):
                obs = {k: obs_grp[k][t] for k in
                       ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object-state")}
                X_list.append(bc.obs_features(obs))
            A_list.append(grp["actions"][:].astype(np.float32))
    return np.asarray(X_list, dtype=np.float32), np.concatenate(A_list, axis=0)


def _forward(self, x):
    return self.net(x)


def train_step(policy, optimizer, xb, ab):
    optimizer.zero_grad()
    loss = F.mse_loss(policy(xb), ab)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_policy(policy, raw_env, n_episodes, horizon, device, seed, mu, sd):
    rng = np.random.default_rng(seed)
    successes = 0
    steps_hist = []
    for ep in range(1, n_episodes + 1):
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        obs = raw_env.reset()
        success = False
        steps = horizon
        for t in range(horizon):
            x = (bc.obs_features(obs) - mu) / (sd + 1e-8)
            with torch.no_grad():
                xt = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                a = policy(xt).squeeze(0).cpu().numpy()
            obs, *_ = raw_env.step(a)
            if raw_env._check_success():
                success = True
                steps = t + 1
                break
        successes += int(success)
        steps_hist.append(steps)
        print(f"[{ep:3d}/{n_episodes}] steps={steps:3d} success={success}")
    print("-" * 60)
    print(f"policy success rate: {successes}/{n_episodes}  mean steps: {np.mean(steps_hist):.1f}")
    return successes, steps_hist


def discriminator_loss(logits_expert, logits_novice):
    loss_e = F.binary_cross_entropy_with_logits(logits_expert, torch.ones_like(logits_expert))
    loss_n = F.binary_cross_entropy_with_logits(logits_novice, torch.zeros_like(logits_novice))
    return loss_e + loss_n


def adversarial_reward(logits):
    return -torch.log(1.0 - torch.sigmoid(logits) + airl.EPS)


def reinforce_update(policy, optimizer, feats, actions, rewards):
    mean = policy(feats)
    logp = torch.distributions.Normal(mean, airl.FIXED_STD).log_prob(actions).sum(dim=-1)
    G = torch.zeros_like(rewards)
    G[-1] = rewards[-1]
    for t in range(len(G) - 2, -1, -1):
        G[t] = rewards[t] + airl.GAMMA * G[t + 1]
    loss = -(logp * G).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def patch_all():
    collect_demos.expert_action = expert_action
    collect_demos.ExpertController.next_action = _next_action
    extract_obs.replay_observation = replay_observation
    bc.build_dataset = build_dataset
    bc.BCNet.forward = _forward
    bc.train_step = train_step
    bc.evaluate_policy = evaluate_policy
    # airl binds `from bc import build_dataset` at import time, so patching
    # bc afterwards would not propagate -- rebind it explicitly (BCNet is a
    # class-attribute patch and propagates on its own).
    airl.build_dataset = build_dataset
    airl.discriminator_loss = discriminator_loss
    airl.adversarial_reward = adversarial_reward
    airl.reinforce_update = reinforce_update


def main():
    parser = argparse.ArgumentParser(description="Week06 instructor solutions runner")
    parser.add_argument("--stage", choices=["collect", "extract", "bc", "airl", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    patch_all()
    stage = args.stage

    if stage in ("collect", "all"):
        print("=" * 70)
        print("STAGE 1: collect scripted-expert demos")
        print("=" * 70)
        sys.argv = [
            "collect_demos.py",
            "--episodes", str(args.episodes),
            "--no-cameras",
            "--seed", str(args.seed),
        ]
        collect_demos.main()

    if stage in ("extract", "all"):
        print("=" * 70)
        print("STAGE 2: replay states to observations")
        print("=" * 70)
        sys.argv = ["extract_obs.py", "--no-cameras"]
        extract_obs.main()

    if stage in ("bc", "all"):
        print("=" * 70)
        print("STAGE 3: behaviour cloning")
        print("=" * 70)
        sys.argv = [
            "bc.py",
            "--epochs", str(args.epochs),
            "--eval-episodes", str(args.eval_episodes),
            "--seed", str(args.seed),
        ]
        bc.main()

    if stage in ("airl", "all"):
        print("=" * 70)
        print("STAGE 4: adversarial IRL (advanced)")
        print("=" * 70)
        sys.argv = [
            "airl.py",
            "--iterations", str(args.iterations),
            "--rollouts", str(args.rollouts),
            "--seed", str(args.seed),
        ]
        airl.main()

    print()
    print("WEEK06_SOLUTIONS_DONE")


if __name__ == "__main__":
    main()
