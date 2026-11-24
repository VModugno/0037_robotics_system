"""COMP0037 2026/27 Coursework 2 — Q3 Starter Code: Track A VLA (nanoVLM + Diffusion Chunk-MPC).

Follows the Week 08 Track A architecture on `MobilePickPlaceEnv` (`PandaOmron`):
  1. `collect_vla_dataset(...)`: collects 64x64 RGB `agentview` frames, 16-D proprioception,
     11-D whole-body actions, and 3 templated mobile-manipulation QA pairs per step.
  2. `TinyViT`, `PixelShuffleProjector`, `NanoVLM`: miniature Vision-Language Model
     trained from scratch with prefix-concat attention masking.
  3. `ConditionalDiffusionHead`: DDPM action-chunking head predicting `chunk_len`-step
     action chunks conditioned on proprioception (+ optional frozen VLM image embeddings),
     executed in closed loop with receding-horizon Chunk-MPC (`exec_steps` k in {1, 2, 4, 8}).

Student TODOs:
  - `TODO(CW2-Q3b)`: `PixelShuffleProjector.forward` (2x2 spatial token compression)
    and `build_prefix_causal_mask` (float additive mask: 0.0 visible, -inf blocked).
  - `TODO(CW2-Q3c)`: `cosine_beta_schedule`, `q_sample`, `ddpm_loss`, and `ddpm_sample`
    for the Conditional Diffusion Action Head + `evaluate_chunk_mpc`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from mobile_pick_place_env import (  # noqa: E402
    STATION_A_BASE,
    STATION_B_BASE,
    extract_proprio,
    make_mobile_pick_place_env,
)
from q1_mobile_lfd import MobilePickPlaceExpert  # noqa: E402


def make_templated_qa(obs: dict, phase: str) -> list[tuple[str, str]]:
    """Generate 3 scene-grounded QA pairs per timestep for nanoVLM training."""
    base_y = float(obs["robot0_base_pos"][1])
    station = "table a" if abs(base_y - STATION_A_BASE[1]) < 0.14 else (
        "table b" if abs(base_y - STATION_B_BASE[1]) < 0.14 else "in the aisle"
    )
    grip_state = "closed" if float(obs["robot0_gripper_qpos"][0]) < 0.025 else "open"
    return [
        ("where is the base ?", station),
        ("is the gripper open or closed ?", grip_state),
        ("what is the robot doing ?", phase.replace("_", " ")),
    ]


def collect_vla_dataset(n_episodes: int = 8, seed: int = 0, img_size: int = 64, horizon: int = 240) -> dict:
    """Collect RGB + proprio + 11-D action + QA dataset using MobilePickPlaceExpert."""
    env = make_mobile_pick_place_env(
        use_camera_obs=True, camera_names="agentview", camera_size=img_size, horizon=horizon, seed=seed
    )
    expert = MobilePickPlaceExpert()
    episodes = []
    attempts = 0
    while len(episodes) < n_episodes and attempts < n_episodes * 3:
        attempts += 1
        obs = env.reset()
        expert.reset()
        imgs, proprios, acts, qas = [], [], [], []
        success = False
        for _ in range(horizon):
            phase = expert.phase
            act = expert.act(obs)
            imgs.append(np.flipud(obs["agentview_image"]).copy())
            proprios.append(extract_proprio(obs))
            acts.append(act)
            qas.append(make_templated_qa(obs, phase))
            obs, _, done, _ = env.step(act)
            if env._check_success():
                success = True
                break
            if done:
                break
        if success:
            episodes.append({
                "images": np.asarray(imgs, dtype=np.uint8),
                "proprio": np.asarray(proprios, dtype=np.float32),
                "actions": np.asarray(acts, dtype=np.float32),
                "qa": qas,
            })
    env.close()
    return {"episodes": episodes, "img_size": img_size}


def build_action_chunks(episodes: list[dict], chunk_len: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (image, proprio, flattened chunk) windows without crossing episode boundaries."""
    imgs, props, chunks = [], [], []
    for ep in episodes:
        T = len(ep["actions"])
        for t in range(T):
            idxs = [min(t + k, T - 1) for k in range(chunk_len)]
            imgs.append(ep["images"][t])
            props.append(ep["proprio"][t])
            chunks.append(ep["actions"][idxs].reshape(-1))
    return (
        np.asarray(imgs, dtype=np.uint8),
        np.asarray(props, dtype=np.float32),
        np.asarray(chunks, dtype=np.float32),
    )


class TinyViT(nn.Module):
    """Lightweight Vision Transformer encoding 64x64 RGB into 64 patch tokens."""

    def __init__(self, img_size: int = 64, patch_size: int = 8, dim: int = 128, depth: int = 2, heads: int = 4):
        super().__init__()
        self.grid = img_size // patch_size
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.randn(1, self.grid * self.grid, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 2, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x).flatten(2).transpose(1, 2) + self.pos
        return self.norm(self.blocks(h))


class PixelShuffleProjector(nn.Module):
    """Compress (B, 64, D_v) patch tokens 2x2 into (B, 16, D_lm) prefix tokens."""

    def __init__(self, vit_dim: int = 128, lm_dim: int = 128, grid: int = 8):
        super().__init__()
        self.grid = grid
        self.proj = nn.Linear(vit_dim * 4, lm_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """TODO(CW2-Q3b): 2x2 pixel-shuffle reshape (B, 64, D) -> (B, 16, 4D) -> Linear."""
        raise NotImplementedError("TODO(CW2-Q3b): implement 2x2 pixel-shuffle projector")


def build_prefix_causal_mask(n_prefix: int, n_text: int, device: torch.device) -> torch.Tensor:
    """TODO(CW2-Q3b): Float additive mask (0.0 visible, -inf blocked).
    Image prefix tokens attend bidirectionally to each other; text tokens attend
    to all image prefix tokens and causally to prior text tokens.
    """
    raise NotImplementedError("TODO(CW2-Q3b): implement prefix-causal float attention mask")


def cosine_beta_schedule(timesteps: int = 20, s: float = 0.008) -> torch.Tensor:
    """TODO(CW2-Q3c): Return cosine DDPM beta schedule tensor of shape (timesteps,)."""
    raise NotImplementedError("TODO(CW2-Q3c): implement cosine DDPM beta schedule")


def evaluate_chunk_mpc(
    policy_fn,
    n_episodes: int = 5,
    chunk_len: int = 4,
    exec_steps: int = 1,
    horizon: int = 240,
    seed: int = 100,
) -> dict:
    """TODO(CW2-Q3c): Closed-loop receding-horizon Chunk-MPC evaluation on MobilePickPlaceEnv."""
    raise NotImplementedError("TODO(CW2-Q3c): implement receding-horizon Chunk-MPC evaluation")


if __name__ == "__main__":
    ds = collect_vla_dataset(n_episodes=1, seed=0, img_size=64)
    ep0 = ds["episodes"][0]
    print(f"VLA dataset self-test: 1 ep, steps={len(ep0['actions'])}, img={ep0['images'].shape}, sample_qa={ep0['qa'][0]}")

