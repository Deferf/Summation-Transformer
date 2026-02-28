"""AdderBoard-compatible submission wrapper.

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

from strict_hardcoded_transformer import StrictHardcodedTransformer, StrictTokenizer


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
    tokenizer = StrictTokenizer()
    model = StrictHardcodedTransformer(tokenizer)
    model.eval()

    wrapped = {
        "tokenizer": tokenizer,
        "model": model,
    }
    metadata = {
        "name": "Strict Hardcoded Pair-Token Adder",
        "author": "andres + codex",
        "params": _count_unique_params(model),
        "architecture": "1L decoder-style transformer, d_model=96, 3 heads, explicit Q/K/V/O + MLP",
        "tricks": [
            "pair-token encoding X[d1,d2]",
            "fixed routing bias in causal attention",
            "full-adder lookup MLP",
            "deterministic hand-coded weights",
        ],
    }
    return wrapped, metadata


def add(model, a: int, b: int) -> int:
    if not (0 <= a <= 9_999_999_999 and 0 <= b <= 9_999_999_999):
        raise ValueError("Inputs must be in [0, 9_999_999_999]")

    with torch.no_grad():
        pred = model["model"].solve(a, b, device="cpu")
    return int(pred)
