"""Week 08 - VLA lab, part 4 (Foundry track): evaluate a real VLA in robosuite.

This script adapts a pretrained VLA from TRI's **VLA Foundry** to the same
robosuite Lift task the rest of the module uses. It is the industrial
counterpart to the tiny VLA you built by hand in parts 1-3: the wiring is
the same (image + instruction -> action chunk), at 1.7B parameters.

VLA Foundry facts that shape this file (verified against the repo):
    * install is git clone + `uv sync` + `uv pip install -e .` -- it is NOT
      on PyPI, requires Python 3.12 and a CUDA GPU with >= 16 GB VRAM, so
      it CANNOT run inside this lab's pixi environment. Run it on Colab
      (A100/L4) or a lab server (see LAB.md, Foundry track);
    * models load via `BaseModel.from_pretrained(...)` (e.g.
      "TRI-ML/Foundry-VLA-1.7B-sim", a diffusion-head VLA: VLM backbone +
      flow-matching action head, 20-D actions, 4 cameras in its training
      distribution);
    * inference is tensor-in/tensor-out through
      `model.generate_actions(input_ids, pixel_values, actions,
      attention_mask, num_inference_steps=..., past_mask=...,
      proprioception=...)` -- `actions` is REQUIRED even at inference
      (it seeds the denoiser shape); the output is NORMALIZED and must be
      denormalized with the processor/normalizer that produced it.

The three junctions below (TODO T11-T13) are exactly where a real
deployment fails: image preprocessing, prompt tokenisation, and action
de-normalisation. Everything else -- the env, the chunked control loop,
success accounting -- is complete and mirrors week 06's evaluator.

Usage (where Foundry is installed; see LAB.md for the Colab recipe):
    python evaluate_vla.py --model TRI-ML/Foundry-VLA-1.7B-sim \
        --instruction "pick up the red cube" --episodes 10
"""

import argparse
import os

import numpy as np

if os.name == "nt":
    os.makedirs("C:\\tmp", exist_ok=True)

try:
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
except ImportError as e:
    raise SystemExit(
        f"[!] robosuite is required even for the Foundry track ({e}).\n"
        "    Run inside the week08 pixi env or pip install -r requirements.txt"
    ) from None

from vlm_data import make_env, state_features  # noqa: E402


# --------------------------------------------------------------------------- #
# the adapter: everything Foundry-specific lives in these three TODOs
# --------------------------------------------------------------------------- #


def make_processor(model_id):
    """Build the tokenizer/normalizer pair that trained the model.

    Foundry checkpoints ship with their processor config; the exact
    constructor depends on the backbone (vision_language_backbone type in
    the model config). Try, in order:
        1. vla_foundry.models.processors.RoboticsProcessor.from_pretrained(model_id)
        2. the AutoProcessor of the backbone's HF repo.
    """
    from vla_foundry.models.processors import RoboticsProcessor  # type: ignore

    return RoboticsProcessor.from_pretrained(model_id)


def encode_observation(processor, obs, instruction, device):
    """TODO T11 (+T12). robosuite obs -> Foundry model inputs.

    Return whatever generate_actions needs for ONE observation: at minimum
    (pixel_values, input_ids, attention_mask). The junctions:
      T11: pixel_values -- resize/crop the agentview image to the ViT's
           expected resolution and normalise with the processor's image
           mean/std (typically ImageNet stats, scale 1/255); layout and
           dtype follow the backbone (check processor.image_processor).
      T12: input_ids -- tokenise `instruction` with the processor's
           tokenizer; pad/trim to a fixed length and build attention_mask.
           (The diffusion VLA needs SOME text; with a vit_backbone the text
           is ignored and zeros are the documented behaviour.)
    """
    raise NotImplementedError("T11/T12: observation -> tensors (see docstring)")


def decode_action(processor, model_output):
    """TODO T13. Normalised model output -> robosuite 7-D OSC_POSE action.

    generate_actions returns normalised actions in the processor's action
    space (the 1.7B models were trained with 20-D actions). You must:
      1. denormalise with the processor's action statistics,
      2. slice/remap to the 7 dims the BASIC composite controller consumes
         ([dx, dy, dz, ax, ay, az, gripper] in [-1, 1]) -- decide and
         document the mapping (e.g. translation/rotation/gripper dims),
      3. clip to [-1, 1].
    If the checkpoint's action space does NOT contain a compatible subset,
    note this in your findings -- that incompatibility is a real deployment
    finding, not a bug in your code.
    """
    raise NotImplementedError("T13: model output -> env action (see docstring)")


# --------------------------------------------------------------------------- #
# control loop (complete)
# --------------------------------------------------------------------------- #


def run_episode(model, processor, env, instruction, device, replan_every, num_inference_steps):
    """One episode with chunked open-loop control (complete).

    The VLA predicts an action chunk per call (like pi0/Foundry); we
    execute `replan_every` steps of it, then re-perceive. replan_every=1 is
    closed-loop (slow, strongest); larger values amortise inference cost.
    """
    obs = env.reset()
    action = np.zeros(7, dtype=np.float64)
    for t in range(env.horizon):
        if t % max(replan_every, 1) == 0:
            model_in = encode_observation(processor, obs, instruction, device)
            if not isinstance(model_in, dict):
                raise SystemExit(
                    "[!] encode_observation must return a dict of tensors "
                    "(pixel_values, input_ids, attention_mask, actions, ...)"
                )
            # generate_actions requires an `actions` tensor even at
            # inference (it seeds the denoiser); T11's docstring puts it
            # in the returned dict, so it flows in with **model_in
            out = model.generate_actions(**model_in, num_inference_steps=num_inference_steps)
            chunk = decode_action(processor, out)
            chunk = np.asarray(chunk, dtype=np.float64).reshape(-1, 7)
            action = chunk[0]
        obs, *_ = env.step(np.clip(action, -1.0, 1.0))
        if env._check_success():
            return True, t + 1
    return False, env.horizon


def main():
    parser = argparse.ArgumentParser(description="Week 08 Foundry VLA evaluation")
    parser.add_argument("--model", type=str, default="TRI-ML/Foundry-VLA-1.7B-sim")
    parser.add_argument("--instruction", type=str, default="pick up the red cube")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--replan-every", type=int, default=4, help="env steps per VLA call")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        from vla_foundry.models.base_model import BaseModel  # noqa: F401
    except ImportError:
        raise SystemExit(
            "[!] vla_foundry is not installed in this environment.\n"
            "    It is NOT on PyPI: git clone https://github.com/TRI-ML/vla_foundry,\n"
            "    cd vla_foundry && uv sync && uv pip install -e .\n"
            "    (Python 3.12, CUDA GPU >= 16 GB VRAM, HF_TOKEN for gated weights).\n"
            "    See LAB.md 'Foundry track' for the Colab recipe."
        ) from None

    import torch

    from vla_foundry.models.base_model import BaseModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} on {device} ...")
    model = BaseModel.from_pretrained(args.model).to(device)
    model.eval()
    processor = make_processor(args.model)

    env = make_env(image_size=args.image_size)
    rng = np.random.default_rng(args.seed)
    successes = []
    for ep in range(args.episodes):
        np.random.seed(int(rng.integers(2**31 - 1)))
        ok, steps = run_episode(
            model, processor, env, args.instruction, device,
            args.replan_every, args.num_inference_steps,
        )
        successes.append(ok)
        print(f"episode {ep + 1:2d}: success={ok} after {steps} steps")
    env.close()
    n = sum(successes)
    print(f"\neval: {n}/{args.episodes} successes ({n / args.episodes:.0%}) "
          f"on instruction {args.instruction!r}")


if __name__ == "__main__":
    main()
