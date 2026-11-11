"""Week 06 - Learning from Demonstration, part 3: Behaviour Cloning.

Behaviour cloning reduces imitation to supervised learning: map observations
to the expert's actions. The expert's P-controller (T1) is essentially a
deterministic function of (eef position, gripper state, cube position), so a
small MLP should clone it well -- that is the hypothesis you will test, and
the failure modes you will measure.

Pipeline: load the observation HDF5 from extract_obs.py -> assemble (X, A)
arrays (T4) -> train an MLP with MSE loss (T5/T6) -> roll out the cloned
policy closed-loop in the simulator and count successes (T7).

Usage:
    pixi run --manifest-path ../pixi.toml -e week06 python bc.py \
        --demos demos/lift_demos_obs.hdf5 --epochs 100 --eval-episodes 10
"""

import argparse
import os
import sys
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from collect_demos import make_env

OBS_DIM = 3 + 4 + 2 + 3  # eef_pos + eef_quat + gripper_qpos + cube_pos = 12
ACTION_DIM = 7


def obs_features(obs):
    """(12,) observation feature vector used everywhere in this week.

    robosuite's Lift `object-state` is (7,) = cube position (3) + quaternion
    (4); the cube position is its first three entries.
    """
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(4),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
            np.asarray(obs["object-state"], dtype=np.float32)[:3],
        ]
    )


# --------------------------------------------------------------------------- #
# TODO T4: dataset assembly
# --------------------------------------------------------------------------- #


def build_dataset(hdf5_path):
    """TODO T4. Read the extracted demo file into flat (X, A) arrays.

    For every demo group `data/demo_*`, stack the per-step observation keys
    (via obs_features) into a (T, 12) array and pair it with the (T, 7)
    actions from the same group. Concatenate all episodes.

    Returns:
        X: (N, 12) float32 observation features.
        A: (N, 7) float32 expert actions.
    """
    raise NotImplementedError("T4: read demos -> (X, A) arrays (see docstring)")


# --------------------------------------------------------------------------- #
# TODO T5: the policy network
# --------------------------------------------------------------------------- #


class BCNet(nn.Module):
    """MLP policy: 12 -> 256 -> 256 -> 7 (tanh output).

    Two hidden layers with ReLU are enough here. The tanh output head matches
    the action range [-1, 1] that OSC_POSE expects.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_DIM),
            nn.Tanh(),
        )

    def forward(self, x):
        """TODO T5: return self.net(x)."""
        raise NotImplementedError("T5: implement BCNet.forward")


# --------------------------------------------------------------------------- #
# TODO T6: one optimisation step
# --------------------------------------------------------------------------- #


def train_step(policy, optimizer, xb, ab):
    """TODO T6. One supervised BC step.

    Zero the gradients, compute the MSE between policy(xb) and ab,
    backpropagate and let the optimizer step. Return the loss as a python
    float (use .item()).
    """
    raise NotImplementedError("T6: implement the BC optimisation step")


# --------------------------------------------------------------------------- #
# TODO T7: closed-loop evaluation
# --------------------------------------------------------------------------- #


def evaluate_policy(policy, raw_env, n_episodes, horizon, device, seed, mu, sd):
    """TODO T7. Roll out the cloned policy and count successes.

    rng = np.random.default_rng(seed); for each episode:
        1. np.random.seed(int(rng.integers(0, 2**31 - 1))) -- robosuite
           randomises object placement from the global numpy RNG.
        2. obs = raw_env.reset()
        3. For each step t in range(horizon):
             x = (obs_features(obs) - mu) / (sd + 1e-8)
             with torch.no_grad(): action = policy(x as a (1, 12) tensor)
             obs, *_ = raw_env.step(action squeezed to (7,) numpy)
             if raw_env._check_success(): mark success and break.
    Print one line per episode and a final summary line.

    Returns:
        (successes, list of steps-per-episode)
    """
    raise NotImplementedError("T7: implement the closed-loop evaluation")


# --------------------------------------------------------------------------- #
# Training driver (complete)
# --------------------------------------------------------------------------- #


def train(X, A, epochs, batch_size, seed, out_path, plot_path):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # standardise observations (actions are already in [-1, 1])
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sd

    # random 90/10 train/val split of steps (fine at this dataset size)
    idx = rng.permutation(len(Xn))
    n_val = max(1, len(Xn) // 10)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr, Atr = Xn[tr_idx], A[tr_idx]
    Xva, Ava = Xn[val_idx], A[val_idx]

    policy = BCNet()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    best_val = float("inf")
    best_state = None
    patience_left = 15
    hist = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(Xtr))
        losses = []
        for start in range(0, len(order), batch_size):
            sel = order[start : start + batch_size]
            xb = torch.as_tensor(Xtr[sel], dtype=torch.float32)
            ab = torch.as_tensor(Atr[sel], dtype=torch.float32)
            losses.append(train_step(policy, optimizer, xb, ab))

        with torch.no_grad():
            xv = torch.as_tensor(Xva, dtype=torch.float32)
            av = torch.as_tensor(Ava, dtype=torch.float32)
            val_loss = nn.functional.mse_loss(policy(xv), av).item()

        tr_loss = float(np.mean(losses)) if losses else float("nan")
        hist["train"].append(tr_loss)
        hist["val"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
            patience_left = 15
        else:
            patience_left -= 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train {tr_loss:.6f}  val {val_loss:.6f}  patience {patience_left}")
        if patience_left == 0:
            print(f"early stop at epoch {epoch} (best val {best_val:.6f})")
            break

    if best_state is not None:
        policy.load_state_dict(best_state)

    payload = {"state_dict": policy.state_dict(), "obs_mean": mu, "obs_std": sd}
    torch.save(payload, out_path)

    plt.figure(figsize=(6, 4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE action loss")
    plt.yscale("log")
    plt.legend()
    plt.title("BC training")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"saved {out_path} and {plot_path}")
    return policy, mu, sd


def evaluate(policy, n_episodes, horizon, device, seed, mu=None, sd=None):
    """Create a fresh camera-less env and delegate to evaluate_policy (T7)."""
    if mu is None or sd is None:
        mu = np.zeros(OBS_DIM)
        sd = np.ones(OBS_DIM)
    raw_env = make_env(cameras=False, horizon=horizon)
    raw_env.reset()
    return evaluate_policy(policy, raw_env, n_episodes, horizon, device, seed, mu, sd)


def main():
    parser = argparse.ArgumentParser(description="Behaviour cloning on Lift demos")
    parser.add_argument("--demos", type=str, default="demos/lift_demos_obs.hdf5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--out", type=str, default="bc_policy.pt")
    parser.add_argument("--plot", type=str, default="bc_training.png")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.demos):
        print(f"[!] demo file not found: {args.demos}")
        print("    run collect_demos.py and extract_obs.py first")
        sys.exit(1)

    try:
        X, A = build_dataset(args.demos)
        policy = BCNet()  # smoke-test the student's forward
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    Complete TODOs T4/T5 in bc.py, then re-run.")
        sys.exit(1)

    print(f"dataset: {X.shape[0]} steps, obs dim {X.shape[1]}, action dim {A.shape[1]}")

    t0 = time.perf_counter()
    try:
        policy, mu, sd = train(
            X, A, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
            out_path=args.out, plot_path=args.plot,
        )
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    Complete TODO T6 in bc.py, then re-run.")
        sys.exit(1)
    print(f"training wall time: {time.perf_counter() - t0:.1f} s")

    if not args.skip_eval:
        print("\n-- closed-loop evaluation (T7) " + "-" * 30)
        try:
            evaluate(policy, args.eval_episodes, args.horizon, args.device, seed=args.seed, mu=mu, sd=sd)
        except NotImplementedError as e:
            print(f"\n[!] {e}")
            print("    Complete TODO T7 in bc.py and re-run with --skip-eval dropped.")


if __name__ == "__main__":
    main()
