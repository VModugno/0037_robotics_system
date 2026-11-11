"""Week 08 - VLA lab, part 5 (Foundry track): fine-tuning scaffold.

This is NOT a training script -- VLA Foundry trains through its own
torchrun entrypoint, and the honest engineering work is everything around
it: dataset conversion, config overrides, launch commands. This scaffold
builds those for the robot data collected in part 1 and prints the exact
pipeline to run on a Foundry machine (Python 3.12, uv, CUDA >= 16 GB).

Foundry facts baked in (verified against the repo):
    * data format: WebDataset tar shards + manifest.jsonl + stats.json
      (converter: vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py
      --type lerobot|spartan --camera_names ... --observation_keys ...);
    * training entrypoint: torchrun --nproc_per_node=N vla_foundry/main.py
      --config_path <yaml> --total_train_samples N
      [--model.resume_from_checkpoint <ckpt> --model.resume_weights_only True];
    * no LoRA/PEFT exists -- freezing is done with config flags:
      vision_language_backbone.freeze, num_vlm_layers_to_use, freeze_text_encoder/
      freeze_image_encoder, --model.freeze;
    * the 100M toy config the tutorials train (vlm_100m.yaml -> then
      vla_diffusion_100m.yaml: vlm_foundry_backbone + 11M diffusion head
      with flow matching) is the documented small-scale path -- use it as
      the override target instead of the 1.7B models.

Usage (anywhere -- it only writes files and prints commands):
    python finetune_vla.py --data vlm_data --out foundry_finetune
    python finetune_vla.py --checklist
"""

import argparse
import json
import os

import numpy as np


PIPELINE_STEPS = [
    ("1. robosuite npz -> LeRobot dataset",
     "convert episodes (frames, states, actions) to LeRobot v2 layout "
     "(episode_index, frame index, observation.images.agentview, "
     "observation.state, action); ~150 lines -- reuse the openpi "
     "examples/robocasa converter as a template, or export via the "
     "LeRobot python API (LeRobotDataset.create(...).add_frame/save_episode)"),
    ("2. LeRobot -> WebDataset tars",
     "python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py "
     "--type lerobot --dataset_path <lerobot_dir> --local_path <tar_dir> "
     '--camera_names "[\'agentview\']" --samples_per_shard 100 '
     "--observation_keys \"['state']\" --action_keys \"['actions']\" "
     "--config_path robotics_preprocessing_params_1past_14future.yaml"),
    ("3. fine-tune from the 1.7B checkpoint",
     "torchrun --nproc_per_node=1 vla_foundry/main.py "
     "--config_path <your overrides yaml (printed below)> "
     "--model.resume_from_checkpoint TRI-ML/Foundry-VLA-1.7B-sim "
     "--model.resume_weights_only True --total_train_samples 5000"),
    ("4. from-scratch toy path (the tutorial scale)",
     "torchrun --nproc_per_node=1 vla_foundry/main.py "
     "--config_path vla_foundry/configs/vla_diffusion_100m.yaml "
     f"--total_train_samples 5000"),
]

OVERRIDE_TEMPLATE = """# Fine-tuning overrides for our robosuite Lift data.
# Start from configs/vla_diffusion_100m.yaml (the tutorial-scale config),
# then override with the values below.
model:
  vision_language_backbone:
    # our images are 64x64 agentview renders; keep the ViT patching but
    # drop num_vlm_layers_to_use to fit a single consumer GPU
    num_vlm_layers_to_use: 4
    freeze: false          # few hundred demos -> freezing helps; try true
  transformer:
    # action head: 7-D OSC_POSE actions, chunk of 4 (match diffusion_action_head.py)
    action_dim: 7
    num_action_head_repeats: 1
  noise_scheduler:
    num_train_timesteps: 1000
    use_flow_matching_scheduler: true   # pi0/Foundry use flow matching
  input_noise_std: 0.1
per_gpu_batch_size: 32                 # first OOM lever: halve this
precision: amp                         # bf16 needs Ampere+; amp on T4
total_train_samples: 5000
"""


def write_overrides(out_dir):
    path = os.path.join(out_dir, "robosuite_lift_overrides.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(OVERRIDE_TEMPLATE)
    return path


def write_dataset_manifest(data_dir, out_dir):
    """Summarise our vlm_data episodes for the conversion step."""
    episodes = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        npz = np.load(os.path.join(data_dir, fname))
        episodes.append(
            {
                "file": fname,
                "steps": int(npz["actions"].shape[0]),
                "action_dim": int(npz["actions"].shape[1]),
                "state_dim": int(npz["states"].shape[1]),
                "image": "64x64x3 agentview (uint8)",
            }
        )
    path = os.path.join(out_dir, "dataset_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "vlm_data.py (robosuite Lift, scripted expert)",
                "episodes": episodes,
                "total_steps": sum(e["steps"] for e in episodes),
                "note": "convert to LeRobot, then preprocess_robotics_to_tar.py --type lerobot",
            },
            f,
            indent=2,
        )
    return path


def main():
    parser = argparse.ArgumentParser(description="Week 08 Foundry fine-tuning scaffold")
    parser.add_argument("--data", type=str, default="vlm_data")
    parser.add_argument("--out", type=str, default="foundry_finetune")
    parser.add_argument("--checklist", action="store_true",
                        help="print the pipeline without touching --data")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not args.checklist:
        if not os.path.isdir(args.data):
            raise SystemExit(f"[!] {args.data} not found -- run vlm_data.py first "
                             "(or use --checklist)")
        manifest = write_dataset_manifest(args.data, args.out)
        print(f"wrote {manifest}")
    overrides = write_overrides(args.out)
    print(f"wrote {overrides}")

    print("\nFoundry fine-tuning pipeline (run on a machine with vla_foundry installed):")
    print("=" * 78)
    for title, body in PIPELINE_STEPS:
        print(f"\n{title}\n  {body}")
    print("=" * 78)
    print("\nReality check before you launch:")
    print("  * vla_foundry needs Python 3.12 + uv (git clone; not on PyPI)")
    print("  * single-GPU VRAM >= 16 GB; per_gpu_batch_size is your OOM lever")
    print("  * no LoRA exists in the repo -- use the freeze flags instead")
    print("  * few hundred demos -> set total_train_samples realistically (~5k)")
    print("  * T4 GPUs: use precision amp (no bf16 on Turing); A100: amp_bf16")


if __name__ == "__main__":
    main()
