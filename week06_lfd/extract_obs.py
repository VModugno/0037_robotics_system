"""Week 06 - Learning from Demonstration, part 2: replay states to observations.

The raw demonstrations produced by `collect_demos.py` (via robosuite's
DataCollectionWrapper) contain only the simulator *states* and the *actions*
-- no images, no proprioception. This is deliberate and standard: states are
the compact ground truth of the episode, and *observations* are a function of
state that can be recomputed at any time. robomimic's
`dataset_states_to_obs.py` does exactly what this script does.

The key mechanic (TODO T3): to get the observation that corresponded to a
recorded state, you (a) push the flattened state back into the simulator,
(b) run `sim.forward()` to recompute all derived quantities (site positions,
camera poses, contacts), and (c) ask the environment for its observation
dict. Do this for every state in every episode and you have a training set.

Output HDF5 layout (robomimic-style, one demo group per episode):

    data/
      demo_0/
        obs/      agentview_image (T,H,W,3) uint8 (only with cameras)
                 robot0_eye_in_hand_image
                 robot0_eef_pos (T,3), robot0_eef_quat (T,4),
                 robot0_gripper_qpos (T,2), robot0_joint_pos (T,7),
                 object-state (T,7)
        actions (T,7)   copied verbatim from the raw file

Usage:
    pixi run --manifest-path ../pixi.toml -e week06 python extract_obs.py \
        --hdf5 demos/lift_demos.hdf5 --output demos/lift_demos_obs.hdf5
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

from collect_demos import make_env

PROP_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "object-state",
)
IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


# --------------------------------------------------------------------------- #
# TODO T3: state -> observation
# --------------------------------------------------------------------------- #


def replay_observation(env, state):
    """TODO T3. Recompute the observation dict for one recorded state.

    Steps:
        1. `env.sim.set_state_from_flattened(state)` -- restores qpos/qvel/act.
        2. `env.sim.forward()` -- recomputes site positions, camera frames and
           every other derived quantity so observations are consistent.
        3. Return `env._get_observations(force_update=True)` -- the same
            internal call robomimic's dataset_states_to_obs.py uses (the
            force_update flag bypasses the observable cache, which would
            otherwise happily serve values from before the state injection);
            robosuite has no public "obs from state" API.

    Args:
        env:   an unwrapped robosuite environment (make_env with collect=False).
        state: (D,) flattened MjSimState as stored by DataCollectionWrapper.

    Returns:
        The observation OrderedDict for this state.
    """
    raise NotImplementedError("T3: implement the state-replay (see docstring)")


# --------------------------------------------------------------------------- #
# Extraction loop (complete)
# --------------------------------------------------------------------------- #


def extract(hdf5_path, output_path, cameras, image_size):
    raw_env = make_env(cameras=cameras, image_size=image_size)
    raw_env.reset()  # context must exist before we can inject states

    with h5py.File(hdf5_path, "r") as fin:
        demos = sorted(fin["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        total_steps = 0
        with h5py.File(output_path, "w") as fout:
            data_grp = fout.create_group("data")
            for name in demos:
                states = fin[f"data/{name}/states"][:]
                actions = fin[f"data/{name}/actions"][:]
                T = len(states)
                obs_store = {k: [] for k in PROP_KEYS}
                if cameras:
                    for k in IMAGE_KEYS:
                        obs_store[k] = []

                for t in range(T):
                    obs = replay_observation(raw_env, states[t])
                    for k in PROP_KEYS:
                        obs_store[k].append(np.asarray(obs[k]))
                    if cameras:
                        for k in IMAGE_KEYS:
                            obs_store[k].append(np.asarray(obs[k]))

                grp = data_grp.create_group(name)
                obs_grp = grp.create_group("obs")
                for k, v in obs_store.items():
                    obs_grp.create_dataset(k, data=np.stack(v))
                grp.create_dataset("actions", data=actions)
                total_steps += T
                print(f"{name}: {T} steps, obs keys: {sorted(obs_store.keys())}")

    print("-" * 60)
    print(f"episodes: {len(demos)}   total steps: {total_steps}")
    print(f"wrote {output_path}")
    return total_steps


def main():
    parser = argparse.ArgumentParser(description="Replay raw demo states to observations")
    parser.add_argument("--hdf5", type=str, default="demos/lift_demos.hdf5")
    parser.add_argument("--output", type=str, default="demos/lift_demos_obs.hdf5")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    if not os.path.exists(args.hdf5):
        print(f"[!] raw demo file not found: {args.hdf5}")
        print("    run collect_demos.py first")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    t0 = time.perf_counter()
    try:
        extract(args.hdf5, args.output, cameras=not args.no_cameras, image_size=args.image_size)
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    Complete TODO T3 (replay_observation) in extract_obs.py, then re-run.")
        sys.exit(1)
    print(f"wall time: {time.perf_counter() - t0:.1f} s")
    print("next: python bc.py --demos demos/lift_demos_obs.hdf5")


if __name__ == "__main__":
    main()
