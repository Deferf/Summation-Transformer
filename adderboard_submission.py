"""AdderBoard-compatible submission wrapper for the 37-parameter spectral model.

Implements the required interface:
  - build_model() -> (model, metadata)
  - add(model, a, b) -> int
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_strictish_h2_d6_singlecarry_from_scratch import ScratchModel, Tokenizer


# Trained 37-parameter spectral model weights (qkvo fixed-basis diagonals + 1-wide MLP).
Q_SCALES = torch.tensor(
    [-0.1265251189, 0.0851078108, 0.0658098981, -0.0625808612, -0.0692180917, 0.0503446385],
    dtype=torch.float32,
)
K_SCALES = torch.tensor(
    [0.0334423110, -0.1416695863, 0.1117569655, -0.0156439319, -0.0523299091, 0.0260327794],
    dtype=torch.float32,
)
V_SCALES = torch.tensor(
    [6.4703655243, 6.5212664604, 6.5689649582, -6.5350785255, -6.5353569984, 6.4883966446],
    dtype=torch.float32,
)
O_SCALES = torch.tensor(
    [6.5905404091, 6.4462933540, 6.4322671890, -6.4658899307, -6.4268512726, 6.5715308189],
    dtype=torch.float32,
)
FC1_W = torch.tensor(
    [[0.0265082344, -0.0029801412, 0.0306621790, 0.0010346355, -5.5169405937, -0.4844546616]],
    dtype=torch.float32,
)
FC1_B = torch.tensor([2.5245070457], dtype=torch.float32)
FC2_W = torch.tensor(
    [[0.0447150506], [-0.0018955168], [0.0449160375], [0.0022844304], [-3.4082012177], [-0.0024604718]],
    dtype=torch.float32,
)


@torch.no_grad()
def _load_trained_37p_weights(model: ScratchModel) -> None:
    model.attn.q_proj.scales.copy_(Q_SCALES)
    model.attn.k_proj.scales.copy_(K_SCALES)
    model.attn.v_proj.scales.copy_(V_SCALES)
    model.attn.o_proj.scales.copy_(O_SCALES)
    model.fc1.weight.copy_(FC1_W)
    model.fc1.bias.copy_(FC1_B)
    model.fc2.weight.copy_(FC2_W)


def _count_unique_params(model: torch.nn.Module) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += p.numel()
    return total


def build_model():
    tokenizer = Tokenizer()
    model = ScratchModel(
        tokenizer,
        init_std=0.0,
        mlp_hidden=1,
        attn_rank=0,
        factorize_o=False,
        mlp_bias_inner=True,
        mlp_bias_outer=False,
        spectral_qkv=True,
        spectral_o=True,
    )
    _load_trained_37p_weights(model)
    model.eval()

    wrapped = {
        "tokenizer": tokenizer,
        "model": model,
    }
    metadata = {
        "name": "Strict-ish Spectral Pair-Token Adder (37 params)",
        "author": "andres + codex",
        "params": _count_unique_params(model),
        "architecture": "1L decoder-style transformer, d_model=6, 2 heads, spectral Q/K/V/O + tiny MLP",
        "tricks": [
            "pair-token encoding X[d1,d2]",
            "fixed routing bias in causal attention",
            "algorithmic output decoding Y[d,carry]",
            "fixed DCT-like basis with trained spectral diagonals",
            "embedded 37 trained weights (no external checkpoint)",
        ],
    }
    return wrapped, metadata


def add(model, a: int, b: int) -> int:
    if not (0 <= a <= 9_999_999_999 and 0 <= b <= 9_999_999_999):
        raise ValueError("Inputs must be in [0, 9_999_999_999]")

    with torch.no_grad():
        pred = model["model"].solve(a, b, device="cpu")
    return int(pred)
