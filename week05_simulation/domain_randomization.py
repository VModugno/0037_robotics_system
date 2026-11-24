"""Week 05 - Part C: domain randomization with batched model fields.

Domain randomization (DR) trains/in evaluates policies across perturbed
physics so they transfer from simulation to the real robot. With MuJoCo
Warp, DR is almost free: the *model itself* can be batched, so every world
gets its own damping and friction in a single step call.

You implement two things:
  T5 - sample per-world randomized parameters
  T6 - roll out all worlds and record the pole angle trajectories

Physics story: every world starts from the SAME state (pole tipped 0.3
rad). Damping and friction differ per world, so trajectories spread.
The spread you measure IS the robustness gap a policy must close.

Run:
    pixi run -e week05 python week05_simulation/domain_randomization.py
    pixi run -e week05 python week05_simulation/domain_randomization.py --nworld 32
"""

import argparse

import numpy as np

try:
    import warp as wp
except ImportError as e:  # pragma: no cover
    raise SystemExit("mujoco-warp needs warp-lang: check `pixi run -e week05` install") from e

import mujoco_warp as mjw

from sim_models import cartpole_model

HINGE_DOF = 1
FLOOR_GEOM = 0


def sample_randomization(rng, nworld):
    """Sample per-world physics perturbations.

    TODO (T5): implement. Return a tuple (hinge_damping, floor_sliding_friction):
      hinge_damping       - np.float32 array (nworld,): log-uniform in
                            [1e-3, 1.0] (log-uniform spends samples across
                            magnitudes, which is what you want for damping)
                            hint: np.exp(rng.uniform(ln(1e-3), ln(1.0), nworld))
      floor_sliding_friction - np.float32 array (nworld,): uniform in [0.2, 1.5]
    """
    raise NotImplementedError("T5: implement sample_randomization")


def build_randomized_model(mjm, nworld, rng):
    """put_model with batched damping/friction fields, then assign samples.

    Complete - read it to see how batched model fields work.
    """
    model = mjw.put_model(
        mjm,
        batch_sizes={"dof_damping": nworld, "geom_friction": nworld},
    )

    hinge_damping, floor_friction = sample_randomization(rng, nworld)

    dampings = np.tile(np.asarray(mjm.dof_damping, dtype=np.float32), (nworld, 1))
    dampings[:, HINGE_DOF] = hinge_damping
    model.dof_damping.assign(dampings)

    frictions = np.tile(np.asarray(mjm.geom_friction, dtype=np.float32), (nworld, 1, 1))
    frictions[:, FLOOR_GEOM, 0] = floor_friction
    model.geom_friction.assign(frictions)
    return model, hinge_damping, floor_friction


def run_randomized_rollout(model, data, n_steps, sample_every=10):
    """Step all worlds for n_steps; record pole angle every sample_every steps.

    TODO (T6): implement. Return a list where entry k is a float32 numpy
    array (nworld,) holding data.qpos[:, 1].numpy() at step (k+1)*sample_every.
    So: for each step, mjw.step(model, data); every sample_every steps append
    a copy of the pole angles. (No synchronize needed between steps - .numpy()
    copies back synchronously.)
    """
    raise NotImplementedError("T6: implement run_randomized_rollout")


def main():
    parser = argparse.ArgumentParser(description="Domain randomization via batched model fields")
    parser.add_argument("--nworld", type=int, default=16, help="number of randomized worlds")
    parser.add_argument("--steps", type=int, default=500, help="rollout steps per world")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    mjm = cartpole_model()
    try:
        device = str(wp.get_device())
    except (RuntimeError, OSError) as e:
        raise SystemExit(
            f"[!] warp runtime failed to initialise:\n    {e}\n"
            "    On managed Windows machines an Application Control policy often\n"
            "    blocks warp\\bin\\warp-clang.dll (WinError 4551). Run this lab on\n"
            "    Colab (see LAB.md) or on a machine where that DLL can load."
        ) from e
    print(f"warp device: {device}")

    try:
        model, hinge_damping, floor_friction = build_randomized_model(mjm, args.nworld, rng)
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    complete T5 in domain_randomization.py, then re-run.")
        return 1

    data = mjw.make_data(mjm, nworld=args.nworld)
    qpos = np.zeros((args.nworld, mjm.nq), dtype=np.float32)
    qpos[:, HINGE_DOF] = 0.3
    data.qpos = wp.array(qpos, dtype=wp.float32)
    mjw.forward(model, data)

    init_angle = data.qpos[:, HINGE_DOF].numpy().copy()

    try:
        traj = run_randomized_rollout(model, data, args.steps, sample_every=10)
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    complete T6 in domain_randomization.py, then re-run.")
        return 1

    # Prepend the true t=0 state so all curves visibly start from the identical initial angle (0.3 rad)
    traj = np.asarray([init_angle] + list(traj))
    final = traj[-1]
    print(f"\ndomain randomization: {args.nworld} worlds, {args.steps} steps, seed {args.seed}")
    print(f"  hinge damping range   : {hinge_damping.min():.4f} .. {hinge_damping.max():.4f}")
    print(f"  floor friction range  : {floor_friction.min():.2f} .. {floor_friction.max():.2f}")
    print(f"  pole angle at start   : mean {traj[0].mean():+.3f} rad, std {traj[0].std():.4f}")
    print(f"  pole angle at the end : mean {final.mean():+.3f} rad, std {final.std():.4f} rad")
    print("  -> same start, different physics: the std is the DR spread your")
    print("     policy must be robust against (record both numbers).")

    plot_trajectories(
        traj,
        args.nworld,
        args.show,
        hinge_damping=hinge_damping,
        floor_friction=floor_friction,
    )
    return 0


def plot_trajectories(traj, nworld, show, hinge_damping=None, floor_friction=None):
    import matplotlib.pyplot as plt

    t = np.arange(traj.shape[0]) * 10 * 0.01  # sample_every * timestep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for w in range(nworld):
        if hinge_damping is not None and floor_friction is not None and nworld <= 8:
            lbl = f"w{w}: d={hinge_damping[w]:.3f}, f={floor_friction[w]:.2f}"
            ax1.plot(t, traj[:, w], lw=1.2, alpha=0.85, label=lbl)
        else:
            ax1.plot(t, traj[:, w], lw=0.8, alpha=0.6)

    if hinge_damping is not None and nworld <= 8:
        ax1.legend(fontsize=7, loc="center right", framealpha=0.85)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("pole angle (rad)")
    ax1.set_title(f"{nworld} randomized worlds, same start")
    ax1.grid(alpha=0.3)

    ax2.hist(traj[-1], bins=min(20, max(8, nworld // 2)), edgecolor="black")
    ax2.set_xlabel("final pole angle (rad)")
    ax2.set_ylabel("worlds")
    ax2.set_title("final-state spread (the DR gap)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = "domain_randomization.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved plot -> {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
