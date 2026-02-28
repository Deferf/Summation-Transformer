from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StrictTokenizer:
    """Tokenizer using pair input tokens X[d1,d2] and output tokens Y[d,carry].

    Sequence (LSD-first):
      [BOS] X0 X1 ... X9 [C0] Y0 Y1 ... Y10 [EOS]

    - Xi encodes the i-th digit pair (a_i, b_i), least significant first.
    - Yi encodes output digit plus carry-out bit.
    """

    PAD: int = 0
    BOS: int = 1
    C0: int = 2
    EOS: int = 3
    X_BASE: int = 4
    Y0_BASE: int = 104
    Y1_BASE: int = 114
    VOCAB_SIZE: int = 124

    def x_id(self, d1: int, d2: int) -> int:
        if not (0 <= d1 <= 9 and 0 <= d2 <= 9):
            raise ValueError(f"X token digits out of range: ({d1}, {d2})")
        return self.X_BASE + 10 * d1 + d2

    def y_id(self, digit: int, carry: int) -> int:
        if not (0 <= digit <= 9 and carry in (0, 1)):
            raise ValueError(f"Invalid y token args: digit={digit}, carry={carry}")
        return (self.Y0_BASE if carry == 0 else self.Y1_BASE) + digit

    def is_y(self, tok: int) -> bool:
        return (self.Y0_BASE <= tok < self.Y0_BASE + 10) or (self.Y1_BASE <= tok < self.Y1_BASE + 10)

    def y_digit(self, tok: int) -> int:
        if self.Y0_BASE <= tok < self.Y0_BASE + 10:
            return tok - self.Y0_BASE
        if self.Y1_BASE <= tok < self.Y1_BASE + 10:
            return tok - self.Y1_BASE
        raise ValueError(f"Not a Y token: {tok}")

    def y_carry(self, tok: int) -> int:
        if self.Y0_BASE <= tok < self.Y0_BASE + 10:
            return 0
        if self.Y1_BASE <= tok < self.Y1_BASE + 10:
            return 1
        raise ValueError(f"Not a Y token: {tok}")

    def encode_problem(self, a: int, b: int) -> List[int]:
        if not (0 <= a < 10**10 and 0 <= b < 10**10):
            raise ValueError("a and b must be in [0, 10^10)")

        a_digits = [int(ch) for ch in reversed(f"{a:010d}")]
        b_digits = [int(ch) for ch in reversed(f"{b:010d}")]
        x_tokens = [self.x_id(a_digits[i], b_digits[i]) for i in range(10)]
        return [self.BOS] + x_tokens + [self.C0]

    def decode_sum(self, generated_tail: List[int]) -> str:
        lsd_digits: List[str] = []
        for tok in generated_tail:
            if tok == self.EOS:
                break
            if self.is_y(tok):
                lsd_digits.append(str(self.y_digit(tok)))
        return "".join(reversed(lsd_digits)) if lsd_digits else ""


class FixedMultiHeadSelfAttention(nn.Module):
    """Fixed-weight causal self-attention with explicit Q/K/V/O matrices.

    Routing is position-based via additive head-specific bias tables.
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int, route_bias: torch.Tensor) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_len = max_len

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer("route_bias", route_bias.clone())

        self._zero_all_weights()

    @torch.no_grad()
    def _zero_all_weights(self) -> None:
        self.q_proj.weight.zero_()
        self.k_proj.weight.zero_()
        self.v_proj.weight.zero_()
        self.o_proj.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + self.route_bias[:, :seq_len, :seq_len].unsqueeze(0)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.o_proj(out)


class StrictHardcodedTransformer(nn.Module):
    """Strict transformer-style hardcoded decoder for exact 10-digit addition.

    Components:
    - token embedding + positional embedding
    - multi-head causal self-attention (explicit Q/K/V/O)
    - MLP (Linear -> ReLU -> Linear)
    - language-model head (vocab logits)

    All weights are fixed and deterministic.
    """

    def __init__(self, tokenizer: StrictTokenizer) -> None:
        super().__init__()
        self.tok = tokenizer

        self.n_heads = 3
        self.d_head = 32
        self.d_model = self.n_heads * self.d_head  # 96

        self.max_len = 24
        self.prefix_len = 12

        # Input feature channels used by token embeddings.
        self.in_d1_base = 0
        self.in_d2_base = 10
        self.in_c_base = 20

        # Attention output channels consumed by the MLP.
        self.h0_base = 0
        self.h1_base = 32
        self.h2_base = 64

        self.token_emb = nn.Embedding(self.tok.VOCAB_SIZE, self.d_model)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)

        route_bias = self._build_route_bias()
        self.attn = FixedMultiHeadSelfAttention(self.d_model, self.n_heads, self.max_len, route_bias=route_bias)

        self.fc1 = nn.Linear(self.d_model, 200, bias=True)
        self.fc2 = nn.Linear(200, self.tok.VOCAB_SIZE, bias=True)

        self._init_token_and_pos_embeddings()
        self._init_attention_value_and_output_projections()
        self._init_mlp_lookup()

    @torch.no_grad()
    def _build_route_bias(self) -> torch.Tensor:
        rb = torch.full((self.n_heads, self.max_len, self.max_len), -20.0)

        for p in range(self.max_len):
            k = p - (self.prefix_len - 1)

            # Head 0 and head 1 read X_k (or BOS on final carry step k=10 -> d1=d2=0).
            if 0 <= k <= 9:
                src_x = 1 + k
            elif k == 10:
                src_x = 0
            else:
                src_x = p

            rb[0, p, src_x] = 20.0
            rb[1, p, src_x] = 20.0

            # Head 2 reads carry from current token (self): C0 at first step, then previous Y token.
            rb[2, p, p] = 20.0

        return rb

    @torch.no_grad()
    def _init_token_and_pos_embeddings(self) -> None:
        self.token_emb.weight.zero_()
        self.pos_emb.weight.zero_()

        # BOS is used as a fixed zero-digit source on the final carry step (k=10).
        self.token_emb.weight[self.tok.BOS, self.in_d1_base + 0] = 1.0
        self.token_emb.weight[self.tok.BOS, self.in_d2_base + 0] = 1.0

        # X[d1,d2] tokens encode both digits in fixed channels.
        for d1 in range(10):
            for d2 in range(10):
                tid = self.tok.x_id(d1, d2)
                self.token_emb.weight[tid, self.in_d1_base + d1] = 1.0
                self.token_emb.weight[tid, self.in_d2_base + d2] = 1.0

        # C0 encodes carry-in = 0.
        self.token_emb.weight[self.tok.C0, self.in_c_base + 0] = 1.0

        # Y[d,c] tokens encode carry-out for next step.
        for d in range(10):
            self.token_emb.weight[self.tok.y_id(d, 0), self.in_c_base + 0] = 1.0
            self.token_emb.weight[self.tok.y_id(d, 1), self.in_c_base + 1] = 1.0

    @torch.no_grad()
    def _init_attention_value_and_output_projections(self) -> None:
        # Q/K remain zeros; routing comes from positional bias.
        self.attn.q_proj.weight.zero_()
        self.attn.k_proj.weight.zero_()

        # V projections:
        # - Head 0 outputs d1 one-hot into channels [0..9] of head0 block.
        # - Head 1 outputs d2 one-hot into channels [0..9] of head1 block.
        # - Head 2 outputs carry one-hot into channels [0..1] of head2 block.
        self.attn.v_proj.weight.zero_()

        for d in range(10):
            self.attn.v_proj.weight[self.h0_base + d, self.in_d1_base + d] = 1.0
            self.attn.v_proj.weight[self.h1_base + d, self.in_d2_base + d] = 1.0

        self.attn.v_proj.weight[self.h2_base + 0, self.in_c_base + 0] = 1.0
        self.attn.v_proj.weight[self.h2_base + 1, self.in_c_base + 1] = 1.0

        # O projection is identity.
        self.attn.o_proj.weight.zero_()
        for i in range(self.d_model):
            self.attn.o_proj.weight[i, i] = 1.0

    @torch.no_grad()
    def _init_mlp_lookup(self) -> None:
        self.fc1.weight.zero_()
        self.fc1.bias.fill_(-2.5)

        idx = 0
        for a in range(10):
            for b in range(10):
                for c in (0, 1):
                    self.fc1.weight[idx, self.h0_base + a] = 1.0
                    self.fc1.weight[idx, self.h1_base + b] = 1.0
                    self.fc1.weight[idx, self.h2_base + c] = 1.0
                    idx += 1

        self.fc2.weight.zero_()
        self.fc2.bias.fill_(-10.0)

        idx = 0
        for a in range(10):
            for b in range(10):
                for c in (0, 1):
                    total = a + b + c
                    digit = total % 10
                    carry = total // 10
                    ytok = self.tok.y_id(digit, carry)
                    self.fc2.weight[ytok, idx] = 20.0
                    idx += 1

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected [B, T], got {tuple(input_ids.shape)}")

        bsz, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Input length {seq_len} exceeds max_len={self.max_len}")

        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        # Single decoder block: attention + residual.
        x = x + self.attn(x)

        # MLP lookup to vocab logits.
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)

        # Force EOS at step k=11 (position p=22 in this fixed layout).
        eos_pos = self.prefix_len - 1 + 11
        if seq_len > eos_pos:
            logits[:, eos_pos, :] = -50.0
            logits[:, eos_pos, self.tok.EOS] = 50.0

        return logits

    @torch.no_grad()
    def generate(self, a: int, b: int, device: torch.device | str = "cpu") -> List[int]:
        dev = torch.device(device)
        seq = self.tok.encode_problem(a, b)

        while len(seq) < self.max_len:
            x = torch.tensor([seq], dtype=torch.long, device=dev)
            logits = self(x)
            next_tok = int(torch.argmax(logits[0, -1]).item())
            seq.append(next_tok)
            if next_tok == self.tok.EOS:
                break

        return seq

    @torch.no_grad()
    def solve(self, a: int, b: int, device: torch.device | str = "cpu") -> str:
        seq = self.generate(a, b, device=device)
        return self.tok.decode_sum(seq[self.prefix_len :])


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def quick_test() -> None:
    tok = StrictTokenizer()
    model = StrictHardcodedTransformer(tok)

    print(f"trainable_params={count_trainable_params(model)}")

    fixed = [
        (0, 0),
        (1, 9),
        (99, 1),
        (1234567890, 9876543210),
        (9999999999, 9999999999),
    ]

    random.seed(2026)
    tests = fixed + [(random.randrange(10**10), random.randrange(10**10)) for _ in range(200)]

    failures = 0
    for a, b in tests:
        pred = model.solve(a, b)
        expected = f"{a + b:011d}"
        if pred != expected:
            failures += 1
            print(f"FAIL a={a:010d} b={b:010d} pred={pred} expected={expected}")
            break

    print(f"tested={len(tests)} failures={failures}")


if __name__ == "__main__":
    quick_test()
