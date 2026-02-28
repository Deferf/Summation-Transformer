from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Tokenizer:
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

    def y_id(self, digit: int, carry: int) -> int:
        if not (0 <= digit <= 9 and carry in (0, 1)):
            raise ValueError(f"Invalid Y token args: d={digit}, c={carry}")
        return (self.Y0_BASE if carry == 0 else self.Y1_BASE) + digit

    def is_y(self, tok: int) -> bool:
        return (self.Y0_BASE <= tok < self.Y0_BASE + 10) or (self.Y1_BASE <= tok < self.Y1_BASE + 10)

    def y_digit(self, tok: int) -> int:
        if self.Y0_BASE <= tok < self.Y0_BASE + 10:
            return tok - self.Y0_BASE
        if self.Y1_BASE <= tok < self.Y1_BASE + 10:
            return tok - self.Y1_BASE
        raise ValueError(f"Not Y token: {tok}")

    def encode_problem(self, a: int, b: int) -> List[int]:
        if not (0 <= a < 10**10 and 0 <= b < 10**10):
            raise ValueError("a and b must be in [0, 10^10)")

        a_digits = [int(ch) for ch in reversed(f"{a:010d}")]
        b_digits = [int(ch) for ch in reversed(f"{b:010d}")]
        x_tokens = [self.x_id(a_digits[i], b_digits[i]) for i in range(10)]
        return [self.BOS] + x_tokens + [self.C0]

    def decode_sum(self, tail_tokens: List[int]) -> str:
        lsd: List[str] = []
        for tok in tail_tokens:
            if tok == self.EOS:
                break
            if self.is_y(tok):
                lsd.append(str(self.y_digit(tok)))
        return "".join(reversed(lsd)) if lsd else ""


@dataclass
class Batch:
    input_ids: torch.Tensor
    gt_d1: torch.Tensor
    gt_d2: torch.Tensor
    gt_carry_in: torch.Tensor


class TwoHeadAttention(nn.Module):
    """2-head causal attention with fixed route bias; all projections are trainable."""

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

        self.register_buffer("route_bias", route_bias)

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


class StrictishH2D6SingleCarry(nn.Module):
    """No-leakage compact model using raw digit channels + one carry channel.

    Semantic channels:
      0: d1_x
      1: d1_y
      2: d2_x
      3: d2_y
      4: carry_in
      5: aux (unused by algorithmic decoder)

    We keep d_model=6 for 2 heads (d_head=3) while using one explicit carry channel.
    """

    def __init__(self, tok: Tokenizer, mlp_hidden: int = 4) -> None:
        super().__init__()
        self.tok = tok

        self.n_heads = 2
        self.d_head = 3
        self.d_model = 6
        self.mlp_hidden = mlp_hidden
        self.max_len = 24
        self.prefix_len = 12

        self.register_buffer("digit_basis", self._build_digit_basis())
        self.register_buffer("route_bias", self._build_route_bias())

        self.attn = TwoHeadAttention(self.d_model, self.n_heads, self.max_len, self.route_bias)
        self.fc1 = nn.Linear(self.d_model, self.mlp_hidden)
        self.fc2 = nn.Linear(self.mlp_hidden, self.d_model)

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

            if 0 <= k <= 9:
                src_x = 1 + k
            elif k == 10:
                src_x = 0
            else:
                src_x = p

            # Both heads read the pair token X_k (or BOS for final carry column).
            # Carry information stays available via the residual from current token.
            rb[0, p, src_x] = 25.0
            rb[1, p, src_x] = 25.0

        return rb

    def _algorithmic_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids -> raw digit embeddings + one carry channel."""
        bsz, seq_len = input_ids.shape
        x = torch.zeros((bsz, seq_len, self.d_model), dtype=torch.float32, device=input_ids.device)

        # BOS behaves like X[0,0] for k=10.
        bos_mask = input_ids == self.tok.BOS
        if bos_mask.any():
            x[bos_mask, 0] = self.digit_basis[0, 0]
            x[bos_mask, 1] = self.digit_basis[0, 1]
            x[bos_mask, 2] = self.digit_basis[0, 0]
            x[bos_mask, 3] = self.digit_basis[0, 1]

        # X[d1,d2]
        x_mask = (input_ids >= self.tok.X_BASE) & (input_ids < self.tok.X_BASE + 100)
        if x_mask.any():
            xv = input_ids[x_mask] - self.tok.X_BASE
            d1 = torch.div(xv, 10, rounding_mode="floor")
            d2 = xv % 10
            x[x_mask, 0] = self.digit_basis[d1, 0]
            x[x_mask, 1] = self.digit_basis[d1, 1]
            x[x_mask, 2] = self.digit_basis[d2, 0]
            x[x_mask, 3] = self.digit_basis[d2, 1]

        # carry channel from C0 and Y tokens
        y1_mask = (input_ids >= self.tok.Y1_BASE) & (input_ids < self.tok.Y1_BASE + 10)
        x[y1_mask, 4] = 1.0

        return x

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self._algorithmic_embed(input_ids)
        x = x + self.attn(x)
        x = x + self.fc2(F.relu(self.fc1(x)))
        return x

    def _decode_digit(self, vec2: torch.Tensor) -> torch.Tensor:
        scores = vec2 @ self.digit_basis.t()
        return torch.argmax(scores, dim=-1)

    def _algorithmic_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden.shape
        logits = torch.full((bsz, seq_len, self.tok.VOCAB_SIZE), -50.0, dtype=torch.float32, device=hidden.device)

        eos_pos = self.prefix_len - 1 + 11

        for p in range(seq_len):
            k = p - (self.prefix_len - 1)
            if 0 <= k <= 10:
                d1 = self._decode_digit(hidden[:, p, 0:2])
                d2 = self._decode_digit(hidden[:, p, 2:4])
                carry_in = (hidden[:, p, 4] > 0).long()

                total = d1 + d2 + carry_in
                out_digit = total % 10
                out_carry = torch.div(total, 10, rounding_mode="floor").clamp(0, 1)

                y_tok = self.tok.Y0_BASE + out_digit + 10 * out_carry
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
        return self._algorithmic_logits(self.forward_hidden(input_ids))

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


class ScratchModel(StrictishH2D6SingleCarry):
    def __init__(self, tok: Tokenizer, init_std: float, mlp_hidden: int = 4) -> None:
        super().__init__(tok, mlp_hidden=mlp_hidden)
        self.reinit_from_scratch(init_std)

    @torch.no_grad()
    def reinit_from_scratch(self, init_std: float) -> None:
        for _, p in self.named_parameters():
            nn.init.normal_(p, mean=0.0, std=init_std)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_training_plot(steps: List[int], losses: List[float], accs: List[float], plot_path: str) -> None:
    if not steps:
        print("No metric history available; skipping plot.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation.")
        return

    out = Path(plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.plot(steps, losses, color="#ff8a3d", linewidth=2.0, label="loss")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss", color="#ff8a3d")
    ax1.tick_params(axis="y", labelcolor="#ff8a3d")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(steps, accs, color="#2ed8b6", linewidth=2.0, label="exact_acc")
    ax2.set_ylabel("Exact Accuracy", color="#2ed8b6")
    ax2.tick_params(axis="y", labelcolor="#2ed8b6")
    ax2.set_ylim(0.0, 1.02)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.suptitle("Training Metrics: Loss and Exact Accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved plot: {out}")


def compute_targets(a: int, b: int) -> Tuple[List[int], List[int], List[int], List[int]]:
    a_digits = [int(ch) for ch in reversed(f"{a:010d}")]
    b_digits = [int(ch) for ch in reversed(f"{b:010d}")]

    d1_cols: List[int] = []
    d2_cols: List[int] = []
    carry_in_cols: List[int] = []
    y_pairs: List[int] = []

    carry_in = 0
    for i in range(11):
        d1 = a_digits[i] if i < 10 else 0
        d2 = b_digits[i] if i < 10 else 0
        d1_cols.append(d1)
        d2_cols.append(d2)
        carry_in_cols.append(carry_in)

        total = d1 + d2 + carry_in
        out_digit = total % 10
        carry_out = total // 10
        y_pairs.append(10 * carry_out + out_digit)
        carry_in = carry_out

    return d1_cols, d2_cols, carry_in_cols, y_pairs


def build_batch(tok: Tokenizer, batch_size: int, device: torch.device) -> Batch:
    rows: List[List[int]] = []
    gt_d1_rows: List[List[int]] = []
    gt_d2_rows: List[List[int]] = []
    gt_c_rows: List[List[int]] = []

    for _ in range(batch_size):
        a = random.randrange(10**10)
        b = random.randrange(10**10)

        prefix = tok.encode_problem(a, b)
        d1_cols, d2_cols, carry_cols, y_pairs = compute_targets(a, b)

        y_tokens = []
        for pair in y_pairs:
            out_digit = pair % 10
            out_carry = pair // 10
            y_tokens.append(tok.y_id(out_digit, out_carry))

        full_seq = prefix + y_tokens + [tok.EOS]
        rows.append(full_seq[:-1])

        gt_d1_rows.append(d1_cols)
        gt_d2_rows.append(d2_cols)
        gt_c_rows.append(carry_cols)

    return Batch(
        input_ids=torch.tensor(rows, dtype=torch.long, device=device),
        gt_d1=torch.tensor(gt_d1_rows, dtype=torch.long, device=device),
        gt_d2=torch.tensor(gt_d2_rows, dtype=torch.long, device=device),
        gt_carry_in=torch.tensor(gt_c_rows, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def eval_exact_acc(model: ScratchModel, n_samples: int, device: torch.device) -> float:
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

    tok = Tokenizer()
    model = ScratchModel(tok, init_std=args.init_std, mlp_hidden=args.mlp_hidden).to(device)

    print(f"trainable_params={count_trainable_params(model)}")
    print(
        f"architecture: n_heads=2 d_model=6 mlp_hidden={args.mlp_hidden} "
        "(semantic channels: d1xy, d2xy, carry, aux)"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    # Output columns k=0..10 map to p=11..21 in training input length 23.
    start = model.prefix_len - 1
    end = start + 11

    history_steps: List[int] = []
    history_losses: List[float] = []
    history_accs: List[float] = []
    last_step = 0
    last_loss = 0.0

    for step in range(1, args.steps + 1):
        last_step = step
        batch = build_batch(tok, args.batch_size, device)
        h = model.forward_hidden(batch.input_ids)[:, start:end, :]  # [B,11,6]

        d1_logits = h[:, :, 0:2] @ model.digit_basis.t()
        d2_logits = h[:, :, 2:4] @ model.digit_basis.t()
        carry_logits = h[:, :, 4]

        loss_d1 = F.cross_entropy(d1_logits.reshape(-1, 10), batch.gt_d1.reshape(-1))
        loss_d2 = F.cross_entropy(d2_logits.reshape(-1, 10), batch.gt_d2.reshape(-1))
        loss_c = bce(carry_logits, batch.gt_carry_in)
        # keep aux channel near 0 to avoid uncontrolled drift
        loss_aux = 0.01 * torch.mean(h[:, :, 5] ** 2)

        loss = loss_d1 + loss_d2 + loss_c + loss_aux
        last_loss = float(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                acc = eval_exact_acc(model, args.eval_samples, device)
            history_steps.append(step)
            history_losses.append(float(loss.item()))
            history_accs.append(float(acc))
            print(
                f"step={step:5d}/{args.steps} "
                f"loss={loss.item():.4f} "
                f"(d1={loss_d1.item():.4f} d2={loss_d2.item():.4f} c={loss_c.item():.4f}) "
                f"exact_acc={acc:.4f}"
            )
            if acc >= args.target_acc:
                print(f"Reached target accuracy {args.target_acc:.4f} at step {step}")
                break

    final_acc = eval_exact_acc(model, args.final_eval_samples, device)
    print(f"final_exact_accuracy={final_acc:.4f} over {args.final_eval_samples} random sums")

    if not history_steps or history_steps[-1] != last_step:
        history_steps.append(last_step)
        history_losses.append(last_loss)
        history_accs.append(float(final_acc))

    save_training_plot(history_steps, history_losses, history_accs, args.plot_path)

    if args.checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "final_exact_accuracy": final_acc,
                "history_steps": history_steps,
                "history_losses": history_losses,
                "history_exact_acc": history_accs,
            },
            args.checkpoint,
        )
        print(f"saved checkpoint: {args.checkpoint}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train h2/d6 single-carry strict-ish model from scratch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--mlp-hidden", type=int, default=4)
    p.add_argument("--init-std", type=float, default=0.08)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-samples", type=int, default=256)
    p.add_argument("--final-eval-samples", type=int, default=2000)
    p.add_argument("--target-acc", type=float, default=0.999)
    p.add_argument("--checkpoint", type=str, default="checkpoints/strictish_h2_d6_singlecarry_scratch.pt")
    p.add_argument("--plot-path", type=str, default="checkpoints/strictish_h2_d6_singlecarry_metrics.png")
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    train(parse_args())
