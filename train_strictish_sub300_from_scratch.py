from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Sub300Tokenizer:
    """Tokenizer using pair input tokens X[d1,d2] and output tokens Y[d,carry].

    Sequence (LSD-first):
      [BOS] X0 X1 ... X9 [C0] Y0 Y1 ... Y10 [EOS]
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

    def y_id(self, d: int, carry: int) -> int:
        if not (0 <= d <= 9 and carry in (0, 1)):
            raise ValueError(f"Invalid Y token args: d={d}, carry={carry}")
        return (self.Y0_BASE if carry == 0 else self.Y1_BASE) + d

    def is_y(self, tok: int) -> bool:
        return (self.Y0_BASE <= tok < self.Y0_BASE + 10) or (self.Y1_BASE <= tok < self.Y1_BASE + 10)

    def y_digit(self, tok: int) -> int:
        if self.Y0_BASE <= tok < self.Y0_BASE + 10:
            return tok - self.Y0_BASE
        if self.Y1_BASE <= tok < self.Y1_BASE + 10:
            return tok - self.Y1_BASE
        raise ValueError(f"Not a Y token: {tok}")

    def encode_problem(self, a: int, b: int) -> List[int]:
        if not (0 <= a < 10**10 and 0 <= b < 10**10):
            raise ValueError("a and b must be in [0, 10^10)")

        a_digits = [int(ch) for ch in reversed(f"{a:010d}")]
        b_digits = [int(ch) for ch in reversed(f"{b:010d}")]
        x_tokens = [self.x_id(a_digits[i], b_digits[i]) for i in range(10)]
        return [self.BOS] + x_tokens + [self.C0]

    def decode_sum(self, tail_tokens: List[int]) -> str:
        lsd_digits: List[str] = []
        for tok in tail_tokens:
            if tok == self.EOS:
                break
            if self.is_y(tok):
                lsd_digits.append(str(self.y_digit(tok)))
        return "".join(reversed(lsd_digits)) if lsd_digits else ""


class TinyFixedAttention(nn.Module):
    """Small multi-head attention with explicit Q/K/V/O projections.

    Routing is enforced by additive position biases (not by token content).
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int, route_bias: torch.Tensor) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer("route_bias", route_bias.clone())
        self._init_fixed_weights()

    @torch.no_grad()
    def _init_fixed_weights(self) -> None:
        self.q_proj.weight.zero_()
        self.k_proj.weight.zero_()
        self.v_proj.weight.zero_()
        self.o_proj.weight.zero_()

        # v_proj: head0 copies d1-angle channels, head1 copies d2-angle channels,
        # head2 copies carry channels.
        # Channel layout: [d1_x, d1_y, d2_x, d2_y, c0, c1]
        self.v_proj.weight[0, 0] = 1.0
        self.v_proj.weight[1, 1] = 1.0
        self.v_proj.weight[2, 2] = 1.0
        self.v_proj.weight[3, 3] = 1.0
        self.v_proj.weight[4, 4] = 1.0
        self.v_proj.weight[5, 5] = 1.0

        # o_proj is identity.
        for i in range(self.d_model):
            self.o_proj.weight[i, i] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + self.route_bias[:, :seq_len, :seq_len].unsqueeze(0)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.o_proj(out)


class StrictishSub300Transformer(nn.Module):
    """Sub-300 parameter, transformer-style hardcoded adder.

    Includes explicit attention and MLP layers but uses algorithmic token mapping
    and algorithmic output decoding to avoid large embedding/output matrices.
    """

    def __init__(self, tokenizer: Sub300Tokenizer) -> None:
        super().__init__()
        self.tok = tokenizer

        self.n_heads = 3
        self.d_head = 2
        self.d_model = self.n_heads * self.d_head  # 6
        self.max_len = 24
        self.prefix_len = 12

        self.register_buffer("digit_basis", self._build_digit_basis())
        self.register_buffer("route_bias", self._build_route_bias())

        self.attn = TinyFixedAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            max_len=self.max_len,
            route_bias=self.route_bias,
        )

        # Tiny MLP block (kept near-neutral by zero init) to preserve transformer structure.
        self.fc1 = nn.Linear(self.d_model, 4, bias=True)
        self.fc2 = nn.Linear(4, self.d_model, bias=True)
        self._init_mlp_neutral()

    @torch.no_grad()
    def _init_mlp_neutral(self) -> None:
        self.fc1.weight.zero_()
        self.fc1.bias.zero_()
        self.fc2.weight.zero_()
        self.fc2.bias.zero_()

    @torch.no_grad()
    def _build_digit_basis(self) -> torch.Tensor:
        vecs = []
        for d in range(10):
            theta = 2.0 * math.pi * d / 10.0
            vecs.append([math.cos(theta), math.sin(theta)])
        return torch.tensor(vecs, dtype=torch.float32)

    @torch.no_grad()
    def _build_route_bias(self) -> torch.Tensor:
        rb = torch.full((self.n_heads, self.max_len, self.max_len), -25.0)

        for p in range(self.max_len):
            k = p - (self.prefix_len - 1)

            # Heads 0 and 1 read X_k (or BOS when k=10, i.e., final carry-only step).
            if 0 <= k <= 9:
                src_x = 1 + k
            elif k == 10:
                src_x = 0
            else:
                src_x = p
            rb[0, p, src_x] = 25.0
            rb[1, p, src_x] = 25.0

            # Head 2 reads current token for carry state (C0 or previous Y).
            rb[2, p, p] = 25.0

        return rb

    def _algorithmic_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids -> 6D features with no learned parameters.

        Layout: [d1_x, d1_y, d2_x, d2_y, c0, c1]
        """
        bsz, seq_len = input_ids.shape
        x = torch.zeros((bsz, seq_len, self.d_model), device=input_ids.device, dtype=torch.float32)

        # BOS acts as zero-digit source for k=10.
        bos_mask = input_ids == self.tok.BOS
        if bos_mask.any():
            x[bos_mask, 0] = self.digit_basis[0, 0]
            x[bos_mask, 1] = self.digit_basis[0, 1]
            x[bos_mask, 2] = self.digit_basis[0, 0]
            x[bos_mask, 3] = self.digit_basis[0, 1]

        # X[d1,d2] tokens.
        x_mask = (input_ids >= self.tok.X_BASE) & (input_ids < self.tok.X_BASE + 100)
        if x_mask.any():
            xv = input_ids[x_mask] - self.tok.X_BASE
            d1 = torch.div(xv, 10, rounding_mode="floor")
            d2 = xv % 10
            x[x_mask, 0] = self.digit_basis[d1, 0]
            x[x_mask, 1] = self.digit_basis[d1, 1]
            x[x_mask, 2] = self.digit_basis[d2, 0]
            x[x_mask, 3] = self.digit_basis[d2, 1]

        # C0 and Y carry states.
        c0_mask = input_ids == self.tok.C0
        x[c0_mask, 4] = 1.0

        y0_mask = (input_ids >= self.tok.Y0_BASE) & (input_ids < self.tok.Y0_BASE + 10)
        y1_mask = (input_ids >= self.tok.Y1_BASE) & (input_ids < self.tok.Y1_BASE + 10)
        x[y0_mask, 4] = 1.0
        x[y1_mask, 5] = 1.0

        return x

    def _decode_digit_from_vec(self, vec: torch.Tensor) -> torch.Tensor:
        """Decode nearest unit-circle digit index from (..., 2) vector."""
        scores = vec @ self.digit_basis.t()  # (..., 10)
        return torch.argmax(scores, dim=-1)

    def _algorithmic_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden.shape
        logits = torch.full((bsz, seq_len, self.tok.VOCAB_SIZE), -50.0, dtype=torch.float32, device=hidden.device)

        eos_pos = self.prefix_len - 1 + 11

        for p in range(seq_len):
            k = p - (self.prefix_len - 1)
            if 0 <= k <= 10:
                d1 = self._decode_digit_from_vec(hidden[:, p, 0:2])
                d2 = self._decode_digit_from_vec(hidden[:, p, 2:4])
                carry = torch.argmax(hidden[:, p, 4:6], dim=-1)

                s = d1 + d2 + carry
                out_d = s % 10
                out_c = torch.div(s, 10, rounding_mode="floor")
                y_tok = self.tok.Y0_BASE + out_d + 10 * out_c
                logits[torch.arange(bsz, device=hidden.device), p, y_tok] = 50.0

            if p == eos_pos:
                logits[:, p, :] = -50.0
                logits[:, p, self.tok.EOS] = 50.0

        return logits

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected [B,T], got {tuple(input_ids.shape)}")
        if input_ids.size(1) > self.max_len:
            raise ValueError(f"Input length {input_ids.size(1)} exceeds max_len={self.max_len}")

        x = self._algorithmic_embed(input_ids)

        # Transformer block (attention + residual + tiny MLP + residual).
        x = x + self.attn(x)
        x = x + self.fc2(F.relu(self.fc1(x)))

        return self._algorithmic_logits(x)

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


@dataclass
class Batch:
    input_ids: torch.Tensor
    gt_d1: torch.Tensor
    gt_d2: torch.Tensor
    gt_carry_in: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ScratchStrictishSub300(StrictishSub300Transformer):
    """Same architecture as StrictishSub300Transformer, but random-initialized."""

    def __init__(self, tokenizer: Sub300Tokenizer, init_std: float = 0.08) -> None:
        super().__init__(tokenizer)
        self.reinit_from_scratch(init_std=init_std)

    @torch.no_grad()
    def reinit_from_scratch(self, init_std: float = 0.08) -> None:
        for _, p in self.named_parameters():
            nn.init.normal_(p, mean=0.0, std=init_std)

        # Keep Q/K small so positional route bias dominates at startup.
        self.attn.q_proj.weight.mul_(0.2)
        self.attn.k_proj.weight.mul_(0.2)

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self._algorithmic_embed(input_ids)
        x = x + self.attn(x)
        x = x + self.fc2(F.relu(self.fc1(x)))
        return x


def _compute_column_targets(a: int, b: int) -> Tuple[List[int], List[int], List[int], List[int]]:
    a_digits = [int(ch) for ch in reversed(f"{a:010d}")]
    b_digits = [int(ch) for ch in reversed(f"{b:010d}")]

    d1_cols = a_digits + [0]
    d2_cols = b_digits + [0]

    carry_in_cols: List[int] = []
    y_tokens_digits: List[int] = []
    y_tokens_carry: List[int] = []

    carry_in = 0
    for i in range(11):
        carry_in_cols.append(carry_in)
        s = d1_cols[i] + d2_cols[i] + carry_in
        out_d = s % 10
        carry_out = s // 10
        y_tokens_digits.append(out_d)
        y_tokens_carry.append(carry_out)
        carry_in = carry_out

    return d1_cols, d2_cols, carry_in_cols, [10 * c + d for d, c in zip(y_tokens_digits, y_tokens_carry)]


def build_batch(tokenizer: Sub300Tokenizer, batch_size: int, device: torch.device) -> Batch:
    input_rows: List[List[int]] = []
    gt_d1_rows: List[List[int]] = []
    gt_d2_rows: List[List[int]] = []
    gt_c_rows: List[List[int]] = []

    for _ in range(batch_size):
        a = random.randrange(10**10)
        b = random.randrange(10**10)

        prefix = tokenizer.encode_problem(a, b)
        d1_cols, d2_cols, carry_in_cols, y_pairs = _compute_column_targets(a, b)

        y_tokens = []
        for pair in y_pairs:
            d = pair % 10
            c = pair // 10
            y_tokens.append(tokenizer.y_id(d, c))

        full_seq = prefix + y_tokens + [tokenizer.EOS]  # length 24
        input_ids = full_seq[:-1]  # length 23

        input_rows.append(input_ids)
        gt_d1_rows.append(d1_cols)
        gt_d2_rows.append(d2_cols)
        gt_c_rows.append(carry_in_cols)

    return Batch(
        input_ids=torch.tensor(input_rows, dtype=torch.long, device=device),
        gt_d1=torch.tensor(gt_d1_rows, dtype=torch.long, device=device),
        gt_d2=torch.tensor(gt_d2_rows, dtype=torch.long, device=device),
        gt_carry_in=torch.tensor(gt_c_rows, dtype=torch.long, device=device),
    )


@torch.no_grad()
def eval_exact_accuracy(model: ScratchStrictishSub300, n_samples: int, device: torch.device) -> float:
    good = 0
    for _ in range(n_samples):
        a = random.randrange(10**10)
        b = random.randrange(10**10)
        pred = model.solve(a, b, device=device)
        exp = f"{a + b:011d}"
        if pred == exp:
            good += 1
    return good / max(1, n_samples)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    tok = Sub300Tokenizer()
    model = ScratchStrictishSub300(tok, init_std=args.init_std).to(device)

    print(f"trainable_params={count_trainable_params(model)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Output positions for k=0..10 in input_ids (length 23): p=11..21.
    start = model.prefix_len - 1
    end = start + 11

    for step in range(1, args.steps + 1):
        batch = build_batch(tok, batch_size=args.batch_size, device=device)

        hidden = model.forward_hidden(batch.input_ids)
        h = hidden[:, start:end, :]  # [B,11,6]

        # Differentiable decode objectives.
        d1_logits = h[:, :, 0:2] @ model.digit_basis.t()  # [B,11,10]
        d2_logits = h[:, :, 2:4] @ model.digit_basis.t()  # [B,11,10]
        c_logits = h[:, :, 4:6]  # [B,11,2]

        loss_d1 = F.cross_entropy(d1_logits.reshape(-1, 10), batch.gt_d1.reshape(-1))
        loss_d2 = F.cross_entropy(d2_logits.reshape(-1, 10), batch.gt_d2.reshape(-1))
        loss_c = F.cross_entropy(c_logits.reshape(-1, 2), batch.gt_carry_in.reshape(-1))
        loss = loss_d1 + loss_d2 + loss_c

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                acc = eval_exact_accuracy(model, n_samples=args.eval_samples, device=device)
            print(
                f"step={step:5d}/{args.steps} "
                f"loss={loss.item():.4f} "
                f"(d1={loss_d1.item():.4f} d2={loss_d2.item():.4f} c={loss_c.item():.4f}) "
                f"exact_acc={acc:.4f}"
            )

            if acc >= args.target_acc:
                print(f"Reached target accuracy {args.target_acc:.4f} at step {step}")
                break

    final_acc = eval_exact_accuracy(model, n_samples=args.final_eval_samples, device=device)
    print(f"final_exact_accuracy={final_acc:.4f} over {args.final_eval_samples} random sums")

    if args.checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "final_exact_accuracy": final_acc,
            },
            args.checkpoint,
        )
        print(f"saved checkpoint: {args.checkpoint}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train strict-ish sub-300 model from random initialization")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--init-std", type=float, default=0.08)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-samples", type=int, default=256)
    p.add_argument("--final-eval-samples", type=int, default=2000)
    p.add_argument("--target-acc", type=float, default=0.999)
    p.add_argument("--checkpoint", type=str, default="checkpoints/strictish_sub300_scratch.pt")
    args, _unknown = p.parse_known_args()
    return args


if __name__ == "__main__":
    train(parse_args())
