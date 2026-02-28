from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PhaseTokenizer:
    """Tokenizer with pair tokens X[d1,d2] and output tokens Y[d,carry].

    Sequence layout (LSD-first):
        [BOS] X0 X1 ... X9 [C0] Y0 Y1 ... Y10 [EOS]

    where Xt = X[a_t, b_t] for least-significant to most-significant digits.
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

    def x_digits(self, tok: int) -> Tuple[int, int]:
        if not (self.X_BASE <= tok < self.X_BASE + 100):
            raise ValueError(f"not an X token: {tok}")
        v = tok - self.X_BASE
        return v // 10, v % 10

    def y_id(self, d: int, carry: int) -> int:
        if not (0 <= d <= 9 and carry in (0, 1)):
            raise ValueError(f"invalid Y token args: d={d}, carry={carry}")
        return (self.Y0_BASE if carry == 0 else self.Y1_BASE) + d

    def is_y(self, tok: int) -> bool:
        return (self.Y0_BASE <= tok < self.Y0_BASE + 10) or (self.Y1_BASE <= tok < self.Y1_BASE + 10)

    def y_digit(self, tok: int) -> int:
        if self.Y0_BASE <= tok < self.Y0_BASE + 10:
            return tok - self.Y0_BASE
        if self.Y1_BASE <= tok < self.Y1_BASE + 10:
            return tok - self.Y1_BASE
        raise ValueError(f"not a Y token: {tok}")

    def y_carry(self, tok: int) -> int:
        if self.Y0_BASE <= tok < self.Y0_BASE + 10:
            return 0
        if self.Y1_BASE <= tok < self.Y1_BASE + 10:
            return 1
        raise ValueError(f"not a Y token: {tok}")

    def encode_problem(self, a: int, b: int) -> List[int]:
        if not (0 <= a < 10**10 and 0 <= b < 10**10):
            raise ValueError("a and b must be in [0, 10^10)")

        a_digits = [int(ch) for ch in reversed(f"{a:010d}")]
        b_digits = [int(ch) for ch in reversed(f"{b:010d}")]

        return [self.BOS] + [self.x_id(a_digits[t], b_digits[t]) for t in range(10)] + [self.C0]

    def decode_sum(self, tokens: List[int]) -> str:
        lsd = []
        for tok in tokens:
            if tok == self.EOS:
                break
            if self.is_y(tok):
                lsd.append(str(self.y_digit(tok)))
        return "".join(reversed(lsd)) if lsd else ""


class MinimalPhaseAdderTransformer(nn.Module):
    """Parameter-minimal decoder-only model using phase arithmetic.

    It uses no trainable parameters. The next-token rule is hardcoded:
    - Read X[d1,d2] for current digit column.
    - Read carry from previous Y token (or C0 for t=0).
    - Map to unit-circle angles with step 2*pi/10.
    - Add angles and detect full-turn crossing to produce next carry.
    - Emit Y[d_out, carry_out].
    """

    def __init__(self, tokenizer: PhaseTokenizer) -> None:
        super().__init__()
        self.tok = tokenizer

        self.prefix_len = 12
        self.max_len = 24

        # Stored as buffers, not trainable params.
        self.register_buffer("two_pi", torch.tensor(2.0 * math.pi))
        self.register_buffer("digit_angle", torch.tensor(2.0 * math.pi / 10.0))

    def _step_target_token(self, seq: torch.Tensor, k: int) -> torch.Tensor:
        """Compute next token for step k for each batch row.

        k=0..10 -> output Y_k
        k=11 -> output EOS
        """
        bsz = seq.size(0)
        out = torch.full((bsz,), self.tok.PAD, dtype=torch.long, device=seq.device)

        if k == 11:
            out.fill_(self.tok.EOS)
            return out

        if not (0 <= k <= 10):
            return out

        if k < 10:
            x_tok = seq[:, 1 + k]
            x_val = x_tok - self.tok.X_BASE
            d1 = torch.div(x_val, 10, rounding_mode="floor")
            d2 = x_val % 10
        else:
            d1 = torch.zeros((bsz,), dtype=torch.long, device=seq.device)
            d2 = torch.zeros((bsz,), dtype=torch.long, device=seq.device)

        if k == 0:
            carry_in = torch.zeros((bsz,), dtype=torch.long, device=seq.device)
        else:
            prev_y = seq[:, self.prefix_len + (k - 1)]
            carry_in = (prev_y >= self.tok.Y1_BASE).long()

        # Phase arithmetic on the unit circle.
        theta = (d1.float() + d2.float() + carry_in.float()) * self.digit_angle
        carry_out = torch.floor(theta / self.two_pi).long().clamp(0, 1)
        theta_mod = theta - carry_out.float() * self.two_pi
        d_out = torch.round(theta_mod / self.digit_angle).long() % 10

        out = self.tok.Y0_BASE + d_out + carry_out * 10
        return out.long()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"expected [B,T], got {tuple(input_ids.shape)}")

        bsz, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"input length {seq_len} exceeds max_len={self.max_len}")

        logits = torch.full((bsz, seq_len, self.tok.VOCAB_SIZE), -50.0, dtype=torch.float32, device=input_ids.device)

        for p in range(seq_len):
            k = p - (self.prefix_len - 1)
            target = self._step_target_token(input_ids, k)
            valid = (k >= 0) and (k <= 11)
            if valid:
                logits[:, p, target] = 50.0

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


def count_buffers(model: nn.Module) -> int:
    return sum(b.numel() for b in model.buffers())


def _demo() -> None:
    tok = PhaseTokenizer()
    model = MinimalPhaseAdderTransformer(tok)

    print(f"trainable_params={count_trainable_params(model)}")
    print(f"buffer_entries={count_buffers(model)}")

    tests = [
        (0, 0),
        (1, 9),
        (99, 1),
        (1234567890, 9876543210),
        (9999999999, 9999999999),
    ]

    random.seed(2026)
    for _ in range(20):
        tests.append((random.randrange(10**10), random.randrange(10**10)))

    fails = 0
    for a, b in tests:
        pred = model.solve(a, b)
        expected = f"{a + b:011d}"
        ok = pred == expected
        if not ok:
            fails += 1
        print(f"{a:010d} + {b:010d} -> {pred} expected={expected} ok={ok}")

    print(f"total={len(tests)} fails={fails}")


if __name__ == "__main__":
    _demo()
