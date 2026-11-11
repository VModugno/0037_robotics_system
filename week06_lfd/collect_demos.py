"""Week 06 - Learning from Demonstration, part 1: collect expert demonstrations.

A hand-written *scripted expert* drives the robosuite `Lift` task (Franka Panda,
OSC_POSE control) through a sequence of end-effector waypoints:

    hover above the cube -> descend to grasp height -> close gripper -> lift.

We do NOT teleoperate. The expert is a simple proportional controller toward
the current waypoint, which is exactly the kind of hand-crafted policy you can
write once you know the task geometry. The point of this lab is the *pipeline*:

    scripted expert -> raw (state, action) demos -> replayed observations
                     -> behaviour cloning -> (advanced) adversarial IRL.

Every episode is recorded with robosuite's `DataCollectionWrapper`, which is the
same recording format used by robomimic and the wider demonstration-collection
ecosystem. Only episodes flagged `successful` survive the merge into HDF5, so a
broken expert silently produces an empty dataset -- watch the console output.

TODO map for this week (files):
    T1  expert_action             (this file)  P-controller for OSC_POSE
    T2  ExpertController         (this file)  waypoint scheduling / phase machine
    T3  replay_observation       extract_obs.py  state -> observation replay
    T4  build_dataset            bc.py       HDF5 -> (features, actions) arrays
    T5  BCNet.forward            bc.py       policy MLP
    T6  train_step               bc.py       one BC optimisation step
    T7  evaluate_policy          bc.py       closed-loop success rate
    T8  discriminator_loss       airl.py     adversarial IRL discriminator
    T9  adversarial_reward       airl.py     reward from discriminator logits
    T10 reinforce_update         airl.py     generator (policy) update

Usage (run from this folder):
    pixi run --manifest-path ../pixi.toml -e week06 python collect_demos.py \
        --episodes 20 --hdf5 demos/lift_demos.hdf5

On Windows the script creates C:\\tmp if it is missing (a known MuJoCo/GL
requirement). On Colab, set MUJOCO_GL=egl *before* importing robosuite.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# --------------------------------------------------------------------------- #
# robosuite import with Windows-friendly error handling
# --------------------------------------------------------------------------- #


def _prepare_windows():
    """Create C:\\tmp (required by MuJoCo's Windows GL bindings) if missing."""
    if os.name == "nt":
        os.makedirs("C:\\tmp", exist_ok=True)


_prepare_windows()

try:
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
    from robosuite.scripts.collect_human_demonstrations import (
        gather_demonstrations_as_hdf5,
    )
    from robosuite.wrappers import DataCollectionWrapper
except Exception as e:  # pragma: no cover - environment problem, not code
    print(f"[!] could not import robosuite: {e}")
    if os.name == "nt" and "mujoco" in str(e).lower():
        print("    Windows hint: make sure C:\\tmp exists; robosuite needs a mujoco")
        print("    <3.5 wheel (data.qM era) with the pypi layout (mujoco.dll inside")
        print("    the mujoco package dir). The pixi environment pins both.")
    raise


# --------------------------------------------------------------------------- #
# Environment construction (shared with extract_obs.py / bc.py / airl.py)
# --------------------------------------------------------------------------- #

CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")


def controller_config():
    """The default BASIC composite controller config (arm: OSC_POSE,
    gripper: GRIP) exactly as robosuite's own collection script loads it."""
    return load_composite_controller_config(controller="BASIC", robot="Panda")


def make_env(cameras=True, image_size=128, horizon=300, control_freq=20, controller=None):
    """Build the Lift environment the same way every script in this lab does.

    Args:
        cameras:      render agentview + eye-in-hand images (True) or run
                      camera-less for speed (False). Camera-less obs has no
                      `{camera}_image` keys -- BC in this lab only needs the
                      proprioceptive keys either way.
        image_size:   square camera resolution in pixels.
        horizon:      episode length in control steps (control_freq 20 Hz).
        control_freq: policy control rate. OSC_POSE needs >= 20 Hz to behave.
        controller:   composite controller config (defaults to BASIC via
                      controller_config()); passed explicitly so the recorded
                      env_info JSON matches the env.

    Notes:
        * lite_physics is robosuite 1.5's default (True) and is passed
          explicitly so the recorded demos are reproducible with this version.
        * ignore_done=True lets the episode run past the moment of success;
          we break out manually once the cube is lifted.
    """
    cfg = controller if controller is not None else controller_config()
    return suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=cameras,
        use_camera_obs=cameras,
        camera_names=list(CAMERA_NAMES) if cameras else None,
        camera_heights=image_size if cameras else 84,
        camera_widths=image_size if cameras else 84,
        control_freq=control_freq,
        horizon=horizon,
        ignore_done=True,
        lite_physics=True,
    )


# --------------------------------------------------------------------------- #
# TODO T1 + T2: the scripted expert
# --------------------------------------------------------------------------- #

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0

PHASES = ("hover", "descend", "grasp", "lift", "hold")


def expert_action(eef_pos, target_pos, grip):
    """TODO T1. One OSC_POSE action moving the gripper toward `target_pos`.

    The default OSC_POSE controller maps each of the first three action
    components from [-1, 1] to a *delta* end-effector displacement of up to
    0.05 m per control step (base frame). A proportional controller therefore
    divides the position error by 0.05 and clips to [-1, 1]:

        a[:3] = clip((target - eef) / 0.05, -1, 1)

    Components 3:6 are a rotation delta (we keep the gripper orientation fixed,
    so zeros) and component 6 is the gripper command (-1 open, +1 close).

    Args:
        eef_pos:    (3,) current end-effector position from obs.
        target_pos: (3,) waypoint to move toward.
        grip:       scalar gripper command, GRIP_OPEN or GRIP_CLOSE.

    Returns:
        (7,) float64 action in [-1, 1].
    """
    raise NotImplementedError("T1: implement the P-controller action (see docstring)")


class ExpertController:
    """TODO T2. A tiny phase machine that selects waypoints for the expert.

    Phases and their targets (cube_pos is the cube centre read from
    obs["object-state"][:3]; it is REFRESHED EVERY STEP -- the cube can be
    nudged mid-approach, and a scripted expert that keeps chasing the
    episode-start position lifts empty-handed):

        hover   : cube + [0, 0, 0.12], gripper open  -> approach from above
        descend : cube + [0, 0, 0.00], gripper open  -> fingertips at cube height
        grasp   : stay at the descend waypoint, gripper closing, for
                  GRIP_STEPS control steps (closing takes time)
        lift    : cube + [0, 0, 0.15], gripper closed -> raises the cube
        hold    : stay at the lift waypoint (kept until the episode ends)

    Move to the next phase when the end-effector is within TOL (0.02 m) of the
    current waypoint, except in `grasp`, which advances on a step counter.
    `next_action(obs)` must return a 7-D OSC_POSE action built with
    `expert_action` toward the current phase's waypoint.
    """

    HOVER_OFFSET = (0.0, 0.0, 0.12)
    DESCEND_OFFSET = (0.0, 0.0, 0.0)
    LIFT_OFFSET = (0.0, 0.0, 0.15)
    TOL = 0.02
    GRIP_STEPS = 40

    def __init__(self, cube_pos):
        self.cube_pos = np.asarray(cube_pos, dtype=np.float64)
        self.phase_idx = 0
        self.grip_count = 0

    # -- helpers you will need ------------------------------------------------

    def _waypoint(self):
        """(3,) target for the current phase."""
        if self.phase_idx == 0:
            return self.cube_pos + np.array(self.HOVER_OFFSET)
        if self.phase_idx == 1:
            return self.cube_pos + np.array(self.DESCEND_OFFSET)
        if self.phase_idx >= 3:
            return self.cube_pos + np.array(self.LIFT_OFFSET)
        return self.cube_pos + np.array(self.DESCEND_OFFSET)

    def _grip(self):
        if self.phase_idx in (0, 1):
            return GRIP_OPEN
        return GRIP_CLOSE

    # -- the TODO -------------------------------------------------------------

    def next_action(self, obs):
        """TODO T2. Return the 7-D action for the current step.

        Steps:
            0. Refresh self.cube_pos from obs["object-state"][:3] (the cube
               may have been nudged since the last step).
            1. Read obs["robot0_eef_pos"].
            2. If within TOL of the current waypoint (or the grasp counter is
               done for the grasp phase), advance self.phase_idx (never past
               the last phase).
            3. In phase `grasp`, increment self.grip_count each step and stay
               put until it reaches GRIP_STEPS.
            4. Return expert_action(eef, current waypoint, current grip).
        """
        raise NotImplementedError("T2: implement the waypoint scheduler (see class docstring)")


# --------------------------------------------------------------------------- #
# Collection loop (complete)
# --------------------------------------------------------------------------- #


def run_expert_episode(env, raw_env, controller, max_steps):
    """Run one episode with the scripted expert; return (steps, success)."""
    obs = env.reset()
    for t in range(max_steps):
        action = controller.next_action(obs)
        obs, _reward, _done, _info = env.step(action)
        if raw_env._check_success():
            return t + 1, True
    return max_steps, False


def collect(n_episodes, directory, hdf5_path, cameras, image_size, horizon, seed):
    """Collect `n_episodes` demonstrations and merge them into one HDF5.

    The DataCollectionWrapper writes each episode to `directory` (state_*.npz
    chunks + model.xml); it flushes on the NEXT reset or at close(). Only
    episodes where the wrapper saw `_check_success()` survive the merge, and
    the merged file uses robomimic's raw layout (data/demo_1/states, actions).
    """
    cfg = controller_config()
    raw_env = make_env(cameras=cameras, image_size=image_size, horizon=horizon, controller=cfg)
    env = DataCollectionWrapper(raw_env, directory)
    rng = np.random.default_rng(seed)
    successes = 0
    steps_hist = []
    for ep in range(n_episodes):
        # robosuite randomises object placement with the *global* numpy RNG at
        # reset; seeding it here makes the episode layout reproducible.
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        obs_probe = env.reset()
        cube_pos = obs_probe["object-state"][:3]
        controller = ExpertController(cube_pos)
        steps, success = run_expert_episode(env, raw_env, controller, horizon)
        successes += int(success)
        steps_hist.append(steps)
        print(
            f"[{ep + 1:3d}/{n_episodes}] steps={steps:3d} "
            f"success={success} cube_start=({cube_pos[0]:.2f}, {cube_pos[1]:.2f}, {cube_pos[2]:.2f})"
        )
    env.close()  # flushes the final episode to disk
    print("-" * 60)
    print(f"expert success rate: {successes}/{n_episodes}")
    if successes == 0:
        print(
            "[!] no successful episodes: gather_demonstrations_as_hdf5 will "
            "produce an EMPTY dataset. Debug your expert (T1/T2) before continuing."
        )
        return 0

    env_info = json.dumps(
        {
            "env_name": "Lift",
            "robots": ["Panda"],
            "controller_configs": cfg,
        }
    )
    out_dir = os.path.dirname(os.path.abspath(hdf5_path))
    gather_demonstrations_as_hdf5(directory, out_dir, env_info)
    gathered = os.path.join(out_dir, "demo.hdf5")
    os.replace(gathered, hdf5_path)
    print(f"merged successful episodes into {hdf5_path}")
    print(f"mean episode length: {np.mean(steps_hist):.1f} steps")
    return successes


def main():
    parser = argparse.ArgumentParser(description="Collect scripted-expert Lift demos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--directory", type=str, default="demos/raw")
    parser.add_argument("--hdf5", type=str, default="demos/lift_demos.hdf5")
    parser.add_argument("--no-cameras", action="store_true", help="run camera-less (faster, no images)")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.directory, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.hdf5)), exist_ok=True)

    t0 = time.perf_counter()
    try:
        n_ok = collect(
            n_episodes=args.episodes,
            directory=args.directory,
            hdf5_path=args.hdf5,
            cameras=not args.no_cameras,
            image_size=args.image_size,
            horizon=args.horizon,
            seed=args.seed,
        )
    except NotImplementedError as e:
        print(f"\n[!] {e}")
        print("    Complete TODOs T1 (expert_action) and T2 (ExpertController)")
        print("    in collect_demos.py, then re-run.")
        sys.exit(1)
    if n_ok:
        print(f"wall time: {time.perf_counter() - t0:.1f} s")
        print("next: python extract_obs.py --hdf5 demos/lift_demos.hdf5")


if __name__ == "__main__":
    main()
