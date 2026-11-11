"""Week 08 - VLA lab, part 2: a nanoVLM-style VLM, from scratch.

This is the advanced arc the module builds toward. You weld together the
same three pieces nanoVLM (huggingface/nanoVLM, v0.1) uses, at teaching
scale, and train them on the robot data you collected with vlm_data.py:

    image --TinyViT--> patch tokens --projector--> VLM prefix
                                                          |
    "Question: ... " --word tokenizer--> text tokens ------+--> TinyLM --> answers

Architecture facts you should recognise from the nanoVLM blog/repo:

    * a ViT **from scratch** (patch conv + learned positions, no CLS token);
    * a **pixel-shuffle projector** that compresses 4 patches into 1 token
      (nanoVLM: 196 patches -> 49 tokens via factor-2 shuffle, then a single
      Linear); ours: 64 -> 16 tokens;
    * **prefix concatenation**: image tokens are prepended to the text
      sequence (v0.1 behaviour) with full attention over the image prefix
      and causal attention over the text;
    * **loss on text positions only** (nanoVLM masks the image prefix with
      ignore_index -100);
    * generation recomputes the whole prefix each step (v0.1 has no KV
      cache) -- fine at this scale;
    * two learning-rate groups: projector fast, towers slow (lr_mp vs
      lr_backbones).

Where we simplify (documented divergences from nanoVLM v0.1): word-level
tokenizer over the templated caption corpus instead of a BPE from SmolLM2;
learned positional embeddings instead of RoPE; LayerNorm instead of
RMSNorm; ~8M parameters instead of 222M. The *wiring* is the same.

TODO map:
    T1  TinyViT.patch_embed       conv + positions
    T2  Projector.forward         pixel shuffle + linear
    T3  NanoVLM.forward_loss      prefix concat, attention mask, CE loss
    T4  NanoVLM.generate          greedy decoding

Usage:
    pixi run --manifest-path ../pixi.toml -e week08 python nanovlm_from_scratch.py \
        --data vlm_data --epochs 15

Then describe held-out frames ("Question: where is the cube?") and read the
self-check in LAB.md.
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

IMG_SIZE = 64          # agentview frames from vlm_data.py
PATCH = 8              # 8x8 = 64 patches
VIT_DIM = 192
VIT_LAYERS = 6
VIT_HEADS = 3
SHUFFLE = 2            # pixel-shuffle factor: 64 patches -> 16 tokens
LM_DIM = 256
LM_LAYERS = 6
LM_HEADS = 4
LR_TOWER = 1e-4         # nanoVLM: lr_backbones
LR_PROJECTOR = 1e-3     # nanoVLM: lr_mp (the projector trains ~10x faster)
MAX_TEXT = 24


# --------------------------------------------------------------------------- #
# word tokenizer (nanoVLM v0.1 uses a SmolLM2 BPE; we go word-level -- the
# templated corpus has a tiny closed vocabulary)
# --------------------------------------------------------------------------- #


class Tokenizer:
    def __init__(self, corpus):
        words = sorted({w for s in corpus for w in s.lower().split()})
        self.itos = ["<pad>", "<bos>", "<eos>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        assert len(self.itos) < 2**16

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi["<bos>"]] + [self.stoi[w] for w in text.lower().split()]

    def decode(self, ids):
        words = [self.itos[i] for i in ids]
        return " ".join(w for w in words if w not in ("<pad>", "<bos>", "<eos>"))


# --------------------------------------------------------------------------- #
# vision tower
# --------------------------------------------------------------------------- #


class ViTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        h = self.n(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.m(x))


class TinyViT(nn.Module):
    """ViT from scratch, siglip-style: patch conv + learned pos, no CLS."""

    def __init__(self, img=IMG_SIZE, patch=PATCH, dim=VIT_DIM, layers=VIT_LAYERS, heads=VIT_HEADS):
        super().__init__()
        self.grid = img // patch
        self.patch = patch
        self.dim = dim
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
        self.blocks = nn.ModuleList([ViTBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def patch_embed(self, x):
        """TODO T1. (B, 3, H, W) float images in [0, 1] -> (B, T, dim) tokens.

        Steps: Conv2d(3, dim, kernel=patch, stride=patch) is NOT built for
        you -- create it in __init__? No: build it here per call is wrong.
        Create `self.proj = nn.Conv2d(...)` in __init__ is the standard
        pattern; do that (add it to __init__), then:
            1. conv the image -> (B, dim, gh, gh),
            2. flatten the spatial grid into (B, T=gh*gh, dim) in ROW-MAJOR
               order (torch.flatten of the last two dims keeps row-major),
            3. add the learned positions `self.pos`,
            4. run the blocks and the final norm.
        """
        raise NotImplementedError("T1: patch embedding (see docstring)")

    def forward(self, x):
        return self.patch_embed(x)


def pixel_shuffle_tokens(x, grid, factor, dim):
    """(B, T, C) tokens in row-major grid -> (B, T//f^2, C*f^2) merged tokens.

    Complete utility (the nanoVLM projector uses einops Rearrange; this is
    the same permutation spelled in torch).
    """
    b, t, c = x.shape
    g = grid
    x = x.view(b, g, g, c)
    x = x.view(b, g // factor, factor, g // factor, factor, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, t // (factor * factor), c * factor * factor)


# --------------------------------------------------------------------------- #
# language tower
# --------------------------------------------------------------------------- #


class LMBlock(nn.Module):
    def __init__(self, dim, heads, max_len):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x, attn_mask):
        h = self.n1(x) + self.pos[:, : x.size(1)]
        x = x + self.attn(
            h, h, h, attn_mask=attn_mask, need_weights=False
        )[0]
        return x + self.mlp(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab, dim=LM_DIM, layers=LM_LAYERS, heads=LM_HEADS, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([LMBlock(dim, heads, max_len) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens, attn_mask):
        """tokens (B, L) -> logits (B, L, vocab); attn_mask (L, L) float additive (0 = visible, -inf = blocked)."""
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.head(self.norm(x))


# --------------------------------------------------------------------------- #
# the VLM (prefix concatenation, nanoVLM v0.1 style)
# --------------------------------------------------------------------------- #


class Projector(nn.Module):
    def __init__(self, vit_dim=VIT_DIM, lm_dim=LM_DIM, factor=SHUFFLE):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(vit_dim * factor * factor, lm_dim, bias=False)  # nanoVLM: bias=False

    def forward(self, patch_tokens):
        """TODO T2. (B, T, vit_dim) -> (B, T/4, lm_dim) VLM prefix tokens.

        Use pixel_shuffle_tokens (complete utility) then self.proj. Two
        lines. This is nanoVLM's ModalityProjector: pixel shuffle squeezes
        spatial detail into channels (196->49 patches at factor 2 there,
        64->16 here), then ONE linear layer maps ViT space to LM space.
        """
        raise NotImplementedError("T2: the projector (see docstring)")


class NanoVLM(nn.Module):
    def __init__(self, tokenizer_len):
        super().__init__()
        self.vit = TinyViT()
        self.projector = Projector()
        self.lm = TinyLM(tokenizer_len)
        self.n_prefix = (IMG_SIZE // PATCH // SHUFFLE) ** 2  # 16

    def vision_tokens(self, images):
        """(B, 3, H, W) -> (B, 16, LM_DIM) (complete)."""
        return self.projector(self.vit(images))

    def forward_loss(self, images, tokens):
        """TODO T3. The v0.1 forward: prefix concat + mask + text-only loss.

        images (B, 3, H, W) float in [0, 1]; tokens (B, L) -- <bos> question
        answer <eos>. Return (loss, n_text_tokens):

            1. image_tokens = self.vision_tokens(images)        (B, P, D)
            2. text_embeddings -- you need token->embedding access on the
               LM: use self.lm.embed(tokens)                     (B, L, D)
            3. concat along the sequence axis: x = [img | text]  (B, P+L, D)
            4. build the attention mask (P+L, P+L) as a FLOAT additive
               mask, 0.0 = visible and -inf = blocked:
                  - every position sees all IMAGE positions (the prefix),
                  - text position i sees text positions <= i (causal),
                  - (equivalently: the mask is causal, except the first P
                    columns are all 0.0).
                torch's MultiheadAttention takes attn_mask of shape
                (L, L) or (N*H, L, L); the (L, L) broadcast form is fine.
                Use a float mask, NOT a bool one: on torch >= 2.x the
                interpretation of a *boolean* attn_mask differs between
                the training and the eval/inference code paths, so a
                bool mask that trains fine can NaN out at eval time.
                A 0.0 / -inf float mask means the same thing everywhere.
            5. run the LM blocks yourself? No -- the blocks live in
               self.lm; replicate their loop here:
                 for blk in self.lm.blocks: x = blk(x, mask)
               then x = self.lm.norm(x), logits = self.lm.head(x).
               (The LM.forward builds its own causal mask; we need the
               prefix-aware mask, so we bypass it.)
            6. loss = cross_entropy on the TEXT positions only:
                 logits_text = logits[:, self.n_prefix:, :]     (B, L, V)
                 targets = tokens shifted by one (predict token t+1 from
                 position t); pad the last position with <pad> so shapes
                 line up. The image prefix contributes NOTHING to the loss
                 -- that is nanoVLM's ignore_index=-100 behaviour.
        """
        raise NotImplementedError("T3: forward + loss (see docstring)")

    @torch.no_grad()
    def generate(self, images, prompt_tokens, max_new=MAX_TEXT):
        """TODO T4. Greedy decoding, v0.1 style (no KV cache: recompute all).

        images (B, 3, H, W); prompt_tokens (B, Lp). For each batch element:
        start the sequence with prompt_tokens and repeatedly
            1. embed the current sequence, prepend the image prefix,
            2. forward with the same prefix-aware mask as T3,
            3. take argmax of the LAST position's logits as the next token,
            4. stop on <eos> (id 2) or after max_new new tokens.
        Return the NEW tokens only: (B, n_new) int tensor (trim each row at
        its <eos>). Use the same block loop as T3 -- factor a private helper
        if you like, but keep forward_loss/generate readable.
        """
        raise NotImplementedError("T4: greedy generation (see docstring)")


# --------------------------------------------------------------------------- #
# data (complete)
# --------------------------------------------------------------------------- #


def load_dataset(data_dir):
    """[(image float (3,H,W), question ids, answer ids)] from vlm_data/."""
    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        npz = np.load(os.path.join(data_dir, fname))
        with open(
            os.path.join(data_dir, fname.replace(".npz", "_captions.json")),
            encoding="utf-8",
        ) as f:
            captions = json.load(f)
        frames = npz["frames"]
        for t, qa in enumerate(captions):
            for key in ("doing", "cube", "gripper"):
                q, a = qa[key]
                samples.append((frames[t], q, a))
    return samples


def build_corpus(samples):
    return [q + " " + a for _, q, a in samples]


# --------------------------------------------------------------------------- #
# training loop (complete)
# --------------------------------------------------------------------------- #


def collate(batch, tok, device):
    imgs = torch.stack(
        [torch.from_numpy(im).permute(2, 0, 1).float() / 255.0 for im, _, _ in batch]
    ).to(device)
    seqs = []
    for _, q, a in batch:
        ids = tok.encode(q) + tok.encode(a)[1:] + [tok.stoi["<eos>"]]
        seqs.append(ids[:MAX_TEXT - 1])
    maxlen = max(len(s) for s in seqs)
    tokens = torch.full((len(seqs), maxlen), tok.stoi["<pad>"], dtype=torch.long)
    for i, s in enumerate(seqs):
        tokens[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return imgs, tokens.to(device)


def train(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    samples = load_dataset(args.data)
    corpus = build_corpus(samples)
    tok = Tokenizer(corpus)
    print(f"dataset: {len(samples)} (image, question, answer) triples; vocab {len(tok)}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(samples))
    n_val = max(1, int(0.1 * len(samples)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    train_s = [samples[i] for i in train_idx]
    val_s = [samples[i] for i in val_idx]

    model = NanoVLM(len(tok)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M parameters "
          f"(ViT {VIT_LAYERS}L/{VIT_DIM}d, LM {LM_LAYERS}L/{LM_DIM}d, "
          f"prefix {model.n_prefix} tokens)")

    # nanoVLM's two param groups: fast projector, slow towers
    optim = torch.optim.AdamW(
        [
            {"params": model.projector.parameters(), "lr": LR_PROJECTOR},
            {"params": list(model.vit.parameters()) + list(model.lm.parameters()),
             "lr": LR_TOWER},
        ]
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_s)
        tot, n = 0.0, 0
        for i in range(0, len(train_s), args.batch_size):
            batch = train_s[i : i + args.batch_size]
            imgs, tokens = collate(batch, tok, device)
            try:
                loss, _ = model.forward_loss(imgs, tokens)
            except NotImplementedError as e:
                print(f"\n[!] {e}")
                print("    Complete TODO T3 in nanovlm_from_scratch.py, then re-run.")
                return None
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item() * len(batch)
            n += len(batch)
        model.eval()
        with torch.no_grad():
            vtot, vn = 0.0, 0
            for i in range(0, len(val_s), args.batch_size):
                batch = val_s[i : i + args.batch_size]
                imgs, tokens = collate(batch, tok, device)
                loss, _ = model.forward_loss(imgs, tokens)
                vtot += loss.item() * len(batch)
                vn += len(batch)
        history.append((tot / max(n, 1), vtot / max(vn, 1)))
        print(f"epoch {epoch:3d}: train {tot / max(n, 1):.4f}  val {vtot / max(vn, 1):.4f}")

    torch.save(
        {"state_dict": model.state_dict(), "vocab": tok.itos, "n_prefix": model.n_prefix},
        args.out,
    )
    print(f"saved -> {args.out}")

    _, ax = plt.subplots(figsize=(6, 4))
    ax.plot([h[0] for h in history], label="train")
    ax.plot([h[1] for h in history], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.legend()
    ax.set_title("nanoVLM training")
    fig = ax.figure
    fig.savefig(args.plot, dpi=120, bbox_inches="tight")
    print(f"plot -> {args.plot}")
    return model, tok


def evaluate(model, tok, args):
    """Held-out QA accuracy: generate answers and compare (exact match)."""
    device = next(model.parameters()).device
    samples = load_dataset(args.data)
    rng = np.random.default_rng(args.seed + 1)
    order = rng.permutation(len(samples))
    n_val = max(1, int(0.1 * len(samples)))
    val_s = [samples[i] for i in order[:n_val]]

    model.eval()
    correct, total = 0, 0
    shown = 0
    for i in range(0, len(val_s), args.batch_size):
        batch = val_s[i : i + args.batch_size]
        imgs, _ = collate(batch, tok, device)
        # generate one prompt at a time: questions differ in length, and
        # right-padding a prompt with <pad> poisons generation (the model
        # never learned to continue after <pad> -- those targets are
        # ignored in training). B=1 avoids padding entirely.
        for j, (im, q, a) in enumerate(batch):
            prompt = torch.tensor(
                tok.encode(q), dtype=torch.long, device=device
            ).unsqueeze(0)
            try:
                out = model.generate(imgs[j : j + 1], prompt)
            except NotImplementedError as e:
                print(f"\n[!] {e}")
                print("    Complete TODO T4 in nanovlm_from_scratch.py, then re-run.")
                return
            pred = tok.decode(out[0].tolist())
            ok = pred.strip() == a
            correct += int(ok)
            total += 1
            if shown < 6:
                print(f"  Q: {q!r:50s} gold {a!r:28s} pred {pred!r} {'OK' if ok else ''}")
                shown += 1
    print(f"\nheld-out exact-match accuracy: {correct}/{total} = {correct / max(total, 1):.1%}")
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Week 08 nanoVLM from scratch")
    parser.add_argument("--data", type=str, default="vlm_data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="nanovlm.pt")
    parser.add_argument("--plot", type=str, default="nanovlm_training.png")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    trained = train(args)
    if trained is None:
        return
    if not args.skip_eval:
        evaluate(trained[0], trained[1], args)


if __name__ == "__main__":
    main()
