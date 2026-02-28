from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HardcodedTokenizer:
    """Tokenizer for fixed-width 10-digit addition in LSD-first order.

    Sequence (token-level):
        [BOS] a0..a9 [PLUS] b0..b9 [C0] y0..y10 [EOS]

    - a0 and b0 are least-significant digits.
    - y_t token encodes both displayed digit and next carry bit.
    - C0 is a special token that means initial carry=0.
    """

    PAD: int = 0
    BOS: int = 1
    PLUS: int = 2
    C0: int = 3
    EOS: int = 4
    DIGIT_BASE: int = 5
    Y0_BASE: int = 15
    Y1_BASE: int = 25
    VOCAB_SIZE: int = 35

    def digit_id(self, d: int) -> int:
        if d < 0 or d > 9:
            raise ValueError(f"digit out of range: {d}")
        return self.DIGIT_BASE + d

    def y_id(self, digit: int, carry: int) -> int:
        if digit < 0 or digit > 9:
            raise ValueError(f"digit out of range: {digit}")
        if carry not in (0, 1):
            raise ValueError(f"carry must be 0 or 1, got {carry}")
        return (self.Y0_BASE if carry == 0 else self.Y1_BASE) + digit

    def is_digit(self, tok: int) -> bool:
        return self.DIGIT_BASE <= tok < self.DIGIT_BASE + 10

    def is_y(self, tok: int) -> bool:
        return self.Y0_BASE <= tok < self.Y0_BASE + 10 or self.Y1_BASE <= tok < self.Y1_BASE + 10

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

        return (
            [self.BOS]
            + [self.digit_id(d) for d in a_digits]
            + [self.PLUS]
            + [self.digit_id(d) for d in b_digits]
            + [self.C0]
        )

    def decode_sum_from_ids(self, ids: List[int]) -> str:
        lsd_digits: List[str] = []
        for tok in ids:
            if tok == self.EOS:
                break
            if self.is_y(tok):
                lsd_digits.append(str(self.y_digit(tok)))

        if len(lsd_digits) == 0:
            return ""
        return "".join(reversed(lsd_digits))


class HardcodedMinimalAdditionTransformer(nn.Module):
    """A hardcoded, decoder-only transformer-like network for exact 10-digit addition.

    Design:
    - 3 fixed routing heads:
      - Head A reads a_t
      - Head B reads b_t
      - Head C reads carry token (self token)
    - Fixed MLP implements full-adder lookup table over (a_t, b_t, carry_t).
    - No training required.
    """

    def __init__(self, tokenizer: HardcodedTokenizer) -> None:
        super().__init__()
        self.tok = tokenizer

        self.prefix_len = 23
        self.max_input_len = 34
        self.sum_predict_start = 22
        self.sum_predict_end = 32
        self.eos_predict_pos = 33

        self.hidden_size = 10 * 10 * 2
        self.feature_size = 22

        self.register_buffer("value_a", torch.zeros(self.tok.VOCAB_SIZE, 10))
        self.register_buffer("value_b", torch.zeros(self.tok.VOCAB_SIZE, 10))
        self.register_buffer("value_c", torch.zeros(self.tok.VOCAB_SIZE, 2))

        self.register_buffer("w1", torch.zeros(self.hidden_size, self.feature_size))
        self.register_buffer("b1", torch.full((self.hidden_size,), -2.5))
        self.register_buffer("w2", torch.zeros(self.tok.VOCAB_SIZE, self.hidden_size))
        self.register_buffer("b2", torch.full((self.tok.VOCAB_SIZE,), -10.0))

        self._init_value_tables()
        self._init_full_adder_mlp()

    def _init_value_tables(self) -> None:
        # Head A: reads a_t digits (and PLUS as a10=0 on the final carry step).
        for d in range(10):
            self.value_a[self.tok.digit_id(d), d] = 1.0
        self.value_a[self.tok.PLUS, 0] = 1.0

        # Head B: reads b_t digits (and C0 as b10=0 on the final carry step).
        for d in range(10):
            self.value_b[self.tok.digit_id(d), d] = 1.0
        self.value_b[self.tok.C0, 0] = 1.0

        # Head C: reads carry from current token.
        self.value_c[self.tok.C0, 0] = 1.0
        for d in range(10):
            self.value_c[self.tok.y_id(d, 0), 0] = 1.0
            self.value_c[self.tok.y_id(d, 1), 1] = 1.0

    def _init_full_adder_mlp(self) -> None:
        # Hidden units represent exact (a_digit, b_digit, carry) matches.
        idx = 0
        for a_digit in range(10):
            for b_digit in range(10):
                for carry in (0, 1):
                    self.w1[idx, a_digit] = 1.0
                    self.w1[idx, 10 + b_digit] = 1.0
                    self.w1[idx, 20 + carry] = 1.0

                    total = a_digit + b_digit + carry
                    out_digit = total % 10
                    next_carry = total // 10
                    self.w2[self.tok.y_id(out_digit, next_carry), idx] = 20.0
                    idx += 1

    def _shift_gather(self, input_ids: torch.Tensor, shift: int) -> torch.Tensor:
        """Gather tokens from position p-shift for each destination position p."""
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        src = pos - shift
        valid = (src >= 0) & (src < seq_len)
        src_clamped = src.clamp(0, seq_len - 1)

        gathered = input_ids.index_select(1, src_clamped)
        pad_fill = torch.full_like(gathered, self.tok.PAD)
        return torch.where(valid.unsqueeze(0), gathered, pad_fill)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input shape [B, T], got {tuple(input_ids.shape)}")

        bsz, seq_len = input_ids.shape
        if seq_len > self.max_input_len:
            raise ValueError(f"Input length {seq_len} exceeds max {self.max_input_len}")

        head_a_ids = self._shift_gather(input_ids, shift=21)
        head_b_ids = self._shift_gather(input_ids, shift=10)
        head_c_ids = self._shift_gather(input_ids, shift=0)

        a_feat = self.value_a[head_a_ids]
        b_feat = self.value_b[head_b_ids]
        c_feat = self.value_c[head_c_ids]

        z = torch.cat([a_feat, b_feat, c_feat], dim=-1)

        pos = torch.arange(seq_len, device=input_ids.device)
        active = ((pos >= self.sum_predict_start) & (pos <= self.sum_predict_end)).float()
        z = z * active.view(1, seq_len, 1)

        h = F.relu(F.linear(z, self.w1, self.b1))
        raw_logits = F.linear(h, self.w2, self.b2)

        logits = torch.full_like(raw_logits, -50.0)
        active_idx = torch.nonzero(active > 0, as_tuple=False).squeeze(-1)
        if active_idx.numel() > 0:
            logits[:, active_idx, :] = raw_logits[:, active_idx, :]

        if seq_len > self.eos_predict_pos:
            logits[:, self.eos_predict_pos, :] = -50.0
            logits[:, self.eos_predict_pos, self.tok.EOS] = 50.0

        return logits

    @torch.no_grad()
    def generate_tokens(self, a: int, b: int, device: torch.device | str = "cpu") -> List[int]:
        device = torch.device(device)
        generated = self.tok.encode_problem(a, b)

        while len(generated) < 35:
            x = torch.tensor([generated], dtype=torch.long, device=device)
            logits = self(x)
            next_tok = int(torch.argmax(logits[0, -1]).item())
            generated.append(next_tok)
            if next_tok == self.tok.EOS:
                break

        return generated

    @torch.no_grad()
    def solve(self, a: int, b: int, device: torch.device | str = "cpu") -> str:
        seq = self.generate_tokens(a, b, device=device)
        return self.tok.decode_sum_from_ids(seq[self.prefix_len :])


if __name__ == "__main__":
    tok = HardcodedTokenizer()
    model = HardcodedMinimalAdditionTransformer(tok)

    tests: List[Tuple[int, int]] = [
        (0, 0),
        (1, 9),
        (99, 1),
        (1234567890, 9876543210),
        (9999999999, 9999999999),
        (3141592653, 2718281828),
    ]

    for a, b in tests:
        pred = model.solve(a, b)
        expected = f"{a + b:011d}"
        ok = pred == expected
        print(f"a={a:010d} b={b:010d} pred={pred} expected={expected} ok={ok}")
