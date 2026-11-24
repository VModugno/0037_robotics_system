"""Week 08 - VLA lab, part 1: collect vision-language-action data.

Renders the robosuite Lift task with the week-06-style scripted expert and
saves, for every control step:

    * the agentview image          (64 x 64 x 3 uint8),
    * 12-D proprio state           (eef pos/quat, gripper qpos, cube pos),
    * the 7-D OSC_POSE action,
    * three templated question/answer pairs -- the language supervision
      for the nanoVLM-style model you build next:

        "what is the robot doing?"  -> phase caption (hover/descend/grasp/lift/hold)
        "where is the cube?"        -> "on the table" / "in the air"
        "is the gripper open or closed?" -> "open" / "closed"

The captions are templated on purpose. This week's learning goal is the
*architecture* -- how a vision encoder, a projector and a language model
are welded into one VLM, and how an action head is attached to it -- not
dataset engineering. Templated captions give every image a clean,
image-grounded label: the model must look (gripper open? cube height?) to
answer correctly, which is exactly the grounding a VLA needs.

Usage:
    pixi run --manifest-path ../pixi.toml -e week08 python vlm_data.py \
        --episodes 10 --directory vlm_data

Output: vlm_data/episode_XXX.npz + vlm_data/meta.json (one entry per
episode with step count and success).
"""

import argparse
import json
import os

import numpy as np

if os.name == "nt":
    os.makedirs("C:\\tmp", exist_ok=True)

import robosuite as suite  # noqa: E402  (needs C:\tmp on Windows first)
from robosuite.controllers import load_composite_controller_config  # noqa: E402
from robosuite.utils import binding_utils
import mujoco

# Defensive compatibility patch: compare MuJoCo joint types as plain ints so robosuite's
# joint-address lookup does not raise an enum-comparison AssertionError (seen on Colab
# when MuJoCo builds are mixed). Behaviour is identical on a clean MuJoCo 3.4 install.
def _safe_get_joint_qpos_addr(self, name):
    joint_id = self.joint_name2id(name)
    joint_type = int(self.jnt_type[joint_id])
    joint_addr = self.jnt_qposadr[joint_id]
    ndim = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else (4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
    return joint_addr if ndim == 1 else (joint_addr, joint_addr + ndim)
binding_utils.MjModel.get_joint_qpos_addr = _safe_get_joint_qpos_addr

def _safe_get_joint_qvel_addr(self, name):
    joint_id = self.joint_name2id(name)
    joint_type = int(self.jnt_type[joint_id])
    joint_addr = self.jnt_dofadr[joint_id]
    ndim = 6 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else (3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
    return joint_addr if ndim == 1 else (joint_addr, joint_addr + ndim)
binding_utils.MjModel.get_joint_qvel_addr = _safe_get_joint_qvel_addr


MAX_DELTA = 0.05
TOL = 0.02
GRIP_STEPS = 30
GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0

# one caption per expert phase -- the vocabulary the VLM must ground
PHASE_CAPTIONS = {
    "hover": "the robot arm moves above the red cube",
    "descend": "the open gripper descends toward the red cube",
    "grasp": "the gripper closes around the red cube",
    "lift": "the robot lifts the red cube above the table",
    "hold": "the robot holds the red cube steady",
}

QA_KEYS = ("doing", "cube", "gripper")


def make_env(image_size=64, horizon=300, control_freq=20, seed=None):
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    if seed is not None:
        np.random.seed(seed)
    return suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=control_freq,
        horizon=horizon,
        ignore_done=True,
        lite_physics=True,
    )


def state_features(obs):
    """The 12-D proprio vector (identical to week 06's bc.py)."""
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            np.asarray(obs["object-state"], dtype=np.float32)[:3],
        ]
    ).astype(np.float32)


def qa_pairs(obs, phase):
    """Three image-grounded question/answer pairs for one step."""
    cube = np.asarray(obs["object-state"], dtype=np.float64)[:3]
    gripper_open = bool(obs["robot0_gripper_qpos"][0] < -0.02)
    cube_where = "on the table" if cube[2] < 0.90 else "in the air"
    return {
        "doing": ("what is the robot doing?", PHASE_CAPTIONS[phase]),
        "cube": ("where is the cube?", cube_where),
        "gripper": ("is the gripper open or closed?", "open" if gripper_open else "closed"),
    }


def run_expert_episode(env, rng):
    """Scripted expert (week 06 pattern): hover -> descend -> grasp -> lift.

    Returns (frames, states, actions, captions, success). `captions[t]` is a
    dict keyed by QA_KEYS.
    """
    obs = env.reset()
    cube = np.asarray(obs["object-state"], dtype=np.float64)[:3]
    phase, grip_steps = "hover", 0
    lift_target = None  # fixed at grasp completion (a moving target never converges)
    frames, states, actions, captions = [], [], [], []

    for t in range(env.horizon):
        frames.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
        states.append(state_features(obs))
        captions.append(qa_pairs(obs, phase))
        cube = np.asarray(obs["object-state"], dtype=np.float64)[:3]  # live tracking
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)

        action = np.zeros(7)
        if phase == "hover":
            target, action[6] = cube + (0, 0, 0.12), GRIP_OPEN
            if np.linalg.norm(target - eef) < TOL:
                phase = "descend"
        elif phase == "descend":
            target, action[6] = cube + (0, 0, 0.0), GRIP_OPEN
            if np.linalg.norm(target - eef) < TOL:
                phase = "grasp"
        elif phase == "grasp":
            target, action[6] = cube, GRIP_CLOSE
            grip_steps += 1
            if grip_steps >= GRIP_STEPS:
                phase = "lift"
                lift_target = cube + (0, 0, 0.15)
        elif phase == "lift":
            target, action[6] = lift_target, GRIP_CLOSE
            if np.linalg.norm(target - eef) < TOL:
                phase = "hold"
        else:  # hold
            target, action[6] = eef, GRIP_CLOSE
            action[:] = 0.0
            action[6] = GRIP_CLOSE

        if phase != "hold":
            action[:3] = np.clip((target - eef) / MAX_DELTA, -1.0, 1.0)
        actions.append(action.astype(np.float32))
        obs, *_ = env.step(action)

        if phase == "hold" and env._check_success():
            break

    success = bool(env._check_success())
    return (
        np.stack(frames),
        np.stack(states),
        np.stack(actions),
        captions,
        success,
    )


def collect(episodes, directory, image_size, horizon, seed):
    os.makedirs(directory, exist_ok=True)
    env = make_env(image_size=image_size, horizon=horizon)
    rng = np.random.default_rng(seed)
    meta = []
    n_success = 0
    for ep in range(episodes):
        np.random.seed(int(rng.integers(2**31 - 1)))
        frames, states, actions, captions, success = run_expert_episode(env, rng)
        fname = os.path.join(directory, f"episode_{ep:03d}.npz")
        np.savez_compressed(
            fname,
            frames=frames,
            states=states,
            actions=actions,
        )
        with open(fname.replace(".npz", "_captions.json"), "w", encoding="utf-8") as f:
            json.dump(captions, f)
        meta.append({"episode": ep, "steps": len(frames), "success": bool(success)})
        n_success += int(success)
        print(f"episode {ep:3d}: {len(frames):3d} steps, success={success}")
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    env.close()
    total = sum(m["steps"] for m in meta)
    print(f"\ncollected {n_success}/{episodes} successful episodes, {total} steps")
    print(f"data -> {directory}/  ({len(QA_KEYS)} QA pairs per step)")
    return n_success, total


def main():
    parser = argparse.ArgumentParser(description="Week 08 VLA data collection")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--directory", type=str, default="vlm_data")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    collect(
        episodes=args.episodes,
        directory=args.directory,
        image_size=args.image_size,
        horizon=args.horizon,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
