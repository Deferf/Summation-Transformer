from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class AdditionTokenizer:
    """Tokenizer for fixed-format addition strings.

    Sequence format:
        ^AAAAAAAAAA+BBBBBBBBBB=CCCCCCCCCCC$
    where A and B are 10-digit zero-padded integers and C is 11-digit zero-padded sum.
    """

    stoi: dict
    itos: dict

    @staticmethod
    def build() -> "AdditionTokenizer":
        vocab = ["^", "$", "+", "="] + [str(d) for d in range(10)]
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = {i: ch for ch, i in stoi.items()}
        return AdditionTokenizer(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def sequence_length(self) -> int:
        # Full sequence length including start/end markers.
        return 35

    @property
    def model_block_size(self) -> int:
        # Causal LM input length: full length minus final token.
        return self.sequence_length - 1

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def format_addition_sample(a: int, b: int) -> str:
    a_text = f"{a:010d}"
    b_text = f"{b:010d}"
    c_text = f"{(a + b):011d}"
    return f"^{a_text}+{b_text}={c_text}$"


def build_train_batch(
    tokenizer: AdditionTokenizer,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate an on-the-fly batch of (input_ids, target_ids)."""
    low = 0
    high = 10**10
    seqs = []
    for _ in range(batch_size):
        a = random.randrange(low, high)
        b = random.randrange(low, high)
        seqs.append(tokenizer.encode(format_addition_sample(a, b)))

    tokens = torch.tensor(seqs, dtype=torch.long, device=device)
    # Teacher-forced next-token targets.
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return x, y


def build_eval_set(
    tokenizer: AdditionTokenizer,
    size: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    rng = random.Random(seed)
    low = 0
    high = 10**10

    seqs = []
    pairs: List[Tuple[int, int]] = []
    for _ in range(size):
        a = rng.randrange(low, high)
        b = rng.randrange(low, high)
        pairs.append((a, b))
        seqs.append(tokenizer.encode(format_addition_sample(a, b)))

    tokens = torch.tensor(seqs, dtype=torch.long, device=device)
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return x, y, pairs
