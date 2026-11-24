# Lab 8 — Vision-Language-Action Models: Build a nanoVLM, Attach a Diffusion Action Head, Meet VLA Foundry

> Week 8 of COMP0037 — VLA models (RT-X, OpenVLA style). In this lab you build the full
> modern-robotics stack **yourself, from scratch, on data you collected yourself**:
> a scripted expert generates images + actions in robosuite; you train a tiny
> Vision-Language Model (nanoVLM-style: ViT → pixel-shuffle projector → LLM prefix);
> you then attach a diffusion action head (π0-style) conditioned on the VLM's image
> embeddings — and watch it beat a proprio-only policy in closed-loop control.
> Track B introduces VLA Foundry (TRI-ML), the industrial version of the same recipe,
> for evaluation and fine-tuning on a real pretrained VLA on Colab/a GPU server.

## 1. Overview and outcomes

A VLA is three modules glued by training, not magic:

```
   image --ViT--> vision tokens --projector--> LLM prefix --action head--> robot actions
   text  -------------------------------------------------^
```

After this lab you can:

- collect a robot dataset with images, states, actions and templated captions (`vlm_data.py`);
- implement a miniature nanoVLM (ViT + pixel-shuffle projector + tiny LM, prefix-concat fusion)
  and train it for visual question answering about the robot's own scene (`nanovlm_from_scratch.py`);
- implement a DDPM action head that emits 4-step action *chunks* conditioned on
  proprioception and/or VLM embeddings, and run it as an MPC-style policy in the loop
  (`diffusion_action_head.py`);
- explain every design choice: why pixel shuffle, why prefix-concat, why diffusion over
  regression, why chunks;
- operate (Track B, GPU) a real pretrained VLA: VLA Foundry's `generate_actions` API
  wrapped for robosuite, and its fine-tuning pipeline.

## 2. Setup

From the `labs/` root:

```
pixi run -e week08 python week08_vla/vlm_data.py --help
```

The `week08` environment has torch + robosuite + mujoco (CPU is enough for track A;
a GPU speeds up training but nothing here *needs* one). On Windows the robosuite
helper already creates `C:\tmp`; if mujoco fails to load, copy `mujoco.dll` from the
`mujoco` package into `robosuite\utils` (see week 6 troubleshooting — same applies here).

Colab (CPU runtime is fine for track A; T4 for speed):

```python
!pip install "numpy<2" torch robosuite==1.5.2 "mujoco>=3.3,<3.5" h5py
!git clone https://github.com/google-deepmind/... # not needed — this lab is self-contained
import os; os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU rendering on Colab
```

Then upload the `week08_vla/` folder and run the scripts with `!python week08_vla/vlm_data.py ...`.

## 3. Files and TODO map

| file | status | what you implement |
|---|---|---|
| `vlm_data.py` | complete | — (read it: week-6-style expert, now with 64×64 images + 3 templated QA pairs per step) |
| `nanovlm_from_scratch.py` | **T1–T4** | patch-embed ViT, pixel-shuffle projector, prefix-concat VLM loss, greedy generation |
| `diffusion_action_head.py` | **T5–T10** | cosine schedule, q_sample, ε-prediction loss, ancestral sampling, windowing, closed-loop eval |
| `evaluate_vla.py` | **T11–T13** | Foundry obs→tensors adapter + action decode (Track B, needs `vla_foundry`, GPU) |
| `finetune_vla.py` | complete | — (run it: it writes the fine-tune config + command you'd execute on a GPU server) |

The TODOs follow the nanoVLM v0.1 recipe (Hugging Face's ~1k-line educational VLM):

- **T1 `TinyViT.patch_embed`** — add `self.proj = nn.Conv2d(3, dim, patch, patch)` to
  `__init__` (the scaffold deliberately omits it), then conv → flatten row-major →
  `+ self.pos` → blocks → norm.
- **T2 `Projector.forward`** — pixel-shuffle the (B, 64, 192) patch grid 2×2 into
  (B, 16, 768) and project with `self.proj` to the LM width. Two lines. The *why*
  is an analysis question: why not just an MLP on all 64 tokens?
- **T3 `NanoVLM.forward_loss`** — concat `[image_prefix | text]`, build the attention
  mask (image prefix sees everything — image tokens never attend *causally*; text is
  causal), cross-entropy on text positions only. **Use a float mask (0 = visible,
  -inf = blocked) — see the footgun note in the docstring and Q4 below.**
- **T4 `NanoVLM.generate`** — greedy decoding with the *same* mask, no KV cache.
- **T5–T8** — the DDPM core on 28-dim flattened action chunks: cosine schedule,
  `q_sample`, ε-prediction MSE, ancestral sampling.
- **T9 `build_windows`** — turn episodes into (state, 4-step chunk) training windows;
  chunks never cross episode boundaries.
- **T10 `evaluate_policy`** — the closed loop: sample a 4-action chunk conditioned
  on the current state (+ VLM embedding of the current frame), execute the first
  action, repeat. That is π0-style receding-horizon control.
- **T11–T13 (Track B)** — the Foundry adapter: image → `pixel_values`, instruction →
  `input_ids`, and crucially the *seed `actions` tensor* `generate_actions` expects
  even at inference, plus denormalization/remapping of the 20-D output down to
  robosuite's 7-D OSC_POSE.

## 4. Experiments

Run each stage in order (all from `week08_vla/`, ~15 minutes total on CPU):

```
pixi run --manifest-path ../pixi.toml -e week08 python vlm_data.py --episodes 10 --directory vlm_data
pixi run --manifest-path ../pixi.toml -e week08 python nanovlm_from_scratch.py --data vlm_data --epochs 15
pixi run --manifest-path ../pixi.toml -e week08 python diffusion_action_head.py --data vlm_data --epochs 150
pixi run --manifest-path ../pixi.toml -e week08 python diffusion_action_head.py --data vlm_data --epochs 150 --vlm-checkpoint nanovlm.pt
```

- **E1 (data)**: 10 expert episodes. Check `meta.json`, one episode's captions, and
  the per-phase caption distribution. Why do we store *three* QA types per frame
  rather than one?
- **E2 (epochs)**: train the diffusion head with `--epochs 30` and then `--epochs 150`.
  Compare closed-loop eval. This is the lab's built-in undertraining lesson: at 30
  epochs the sampler *looks* converged (val ε-MSE ≈ 0.07) yet executes 0/10.
- **E3 (ablation)**: proprio-only vs `--vlm-checkpoint nanovlm.pt`. The headline result:
  does a VLM that scores 47.6% on VQA still help control? (Spoiler: yes.)
- **E4 (VLM honesty)**: compute the majority-class baseline for each QA type (we give
  ours below). Your VLM *underperforms* predicting the most common answer — explain
  what that says about vision grounding vs language priors at this scale.
- **E5 (mask footgun)**: switch `_prefix_mask` to a bool mask (`True` = visible) and
  re-run training. Watch train loss fall but validation go `nan` every epoch —
  investigate until you can explain it in one paragraph (Q4).
- **E6 (Track B, GPU)**: on a Colab A100/16GB+ server with `vla_foundry` installed,
  run `evaluate_vla.py` with `TRI-ML/Foundry-VLA-1.7B-sim`, then `finetune_vla.py`
  to generate the fine-tune plan. Report what worked and what you had to adapt.

## 5. Self-check — reference numbers (seed 0, CPU, robosuite Lift)

Your numbers should be in this ballpark (exact values depend on torch version):

```
# vlm_data.py
collected 10/10 successful episodes, total steps 626   # 61-65 steps per episode

# nanovlm_from_scratch.py (15 epochs)
model: 7.9M parameters
epoch  1: train ce 1.2004 | val ce 0.3736
epoch 15: train ce 0.0018 | val ce 0.0018
held-out exact-match accuracy: 89/187 = 47.6%
  per type: gripper 74/74 (100%) | doing 10/47 (21.3%) | cube 5/66 (7.6%)
  majority baselines: closed 74/74 | 'closes around the red cube' 21/47 | 'on the table' 61/66
  → majority-everything baseline: 156/187 = 83.4%

# diffusion_action_head.py, proprio-only, 150 epochs
final val eps-MSE ~0.03 (30-epoch run bottoms at 0.069 and FAILS closed loop)
eval: 8/10 successes, mean steps 101.8        # successes 38-68 steps, 2 stalls at 300

# diffusion_action_head.py --vlm-checkpoint nanovlm.pt, 150 epochs
conditioning: VLM embedding (256-D) + proprio (12-D)
eval: 10/10 successes, mean steps 80.5        # 51-151 steps
```

If your VLM evaluates to ~0% with empty predictions, re-read the evaluate() docstring:
prompts are generated **one at a time (B=1)**. Batch-generation with right-padded
prompts produces only `<eos>` — the model never saw a target after `<pad>`.

## 6. Deliverables & Discussion Questions

Record your experimental findings and answer the following questions in your lab notes:

**Deliverables checklist**

- [ ] `meta.json` + 1 caption file excerpt + per-phase caption counts (E1)
- [ ] nanovlm training plot (train/val CE) + accuracy table incl. majority baselines
- [ ] 30-vs-150-epoch diffusion comparison table (E2)
- [ ] proprio vs +VLM eval table (E3)
- [ ] E5 footgun write-up if you did it
- [ ] commands + seeds + machine + wall-times for your runs

Questions:

1. **Pixel shuffle**: why does nanoVLM shuffle 2×2 patch groups and project them
   (64→16 tokens) instead of feeding all 64 tokens to the LM, or mean-pooling?
   Cost/benefit in tokens and spatial detail.
2. **Prefix-concat vs cross-attention**: nanoVLM v0.1 concatenates image tokens as a
   prefix under one causal mask. Flamingo/RT-2 style use cross-attention. Compare the
   two fusion styles for (a) training data efficiency, (b) inference cost.
3. **Text-only loss**: nanoVLM trains the LM only on answer tokens (images are never
   text targets). When does this become a problem (think: longer multi-turn dialogue,
   captioning instead of QA)?
4. **The bool-mask footgun (E5)**: you observed train CE finite, val CE `nan`, with
   the *identical* batch. Explain what `nn.MultiheadAttention` does with a bool
   mask in each code path and why a float 0/-inf mask fixes it. This bit real
   practitioners on real torch versions.
5. **Chunks**: why predict 4 actions and execute 1 (MPC) instead of 1 action per
   inference, or 300 per inference? What does the action-chunk horizon trade off?
6. **DDPM vs regression**: the BC network of week 6 predicts the mean action and
   fails on multimodal data. Explain why the diffusion head handles multimodality,
   and how flow matching (Foundry's choice) achieves the same without 100 sampling
   steps.
7. **The ablation (E3)**: the VLM scores *below* the majority-class baseline on VQA,
   yet its embeddings lift the policy from 8/10 to 10/10. Reconcile: what does the
   action head use from the VLM that exact-match QA accuracy doesn't measure?
8. **Track B (if attempted)**: document your Foundry adapter experience — the seed
   `actions` tensor, the 20-D → 7-D remap, and whether the sim checkpoint's action
   space admits a compatible subset. An honest "incompatible, here's why" is a
   full-credit deployment finding.

## 7. Going further

- Make the VQA *vision-forced*: ask "is the cube left or right of the table centre?"
  (labels from the x coordinate, not the phase) — does accuracy beat majority now?
- Add KV caching to `generate`; measure the speedup at prefix 16.
- Swap the DDPM for a flow-matching head (like Foundry): linear β interpolation
  between noise and data, fewer integration steps. Compare sample quality at equal
  epochs.
- Fine-tune *your* VLM embeddings end-to-end with the diffusion loss (unfreeze the
  projector) — the full VLA recipe in miniature.
- Track B deep-dive: run Foundry's `tutorials/training_llm_vlm_vla.ipynb`
  (LLM → VLM → VLA from scratch, ~100M configs) on a 16GB GPU and relate each yaml
  block to the module you built by hand.

## 8. References

- nanoVLM (Hugging Face blog + repo, v0.1): the architecture this lab miniaturizes.
- Ho et al., *Denoising Diffusion Probabilistic Models* (2020); Nichol & Dhariya,
  *Improved DDPM* (2021) — cosine schedule.
- Chi et al., *Diffusion Policy* (2023) — action chunks + receding horizon.
- Black et al., *π0* (2024) — VLM + flow-matching action head; the exact stack you built.
- TRI-ML *VLA Foundry* + arXiv 2604.19728 — the industrial pipeline (Track B);
  `Foundry-VLA-1.7B-sim` model card on Hugging Face.
- OpenVLA / RT-X papers for the general VLA landscape.
