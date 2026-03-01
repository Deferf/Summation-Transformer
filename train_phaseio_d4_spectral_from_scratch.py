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
    gt_out_digit: torch.Tensor
    gt_out_carry: torch.Tensor


class SpectralLinearFixedU(nn.Module):
    """Square linear map with fixed orthogonal basis and trainable diagonal only."""

    def __init__(self, d_model: int, basis: torch.Tensor) -> None:
        super().__init__()
        if basis.shape != (d_model, d_model):
            raise ValueError(f"basis must be [{d_model},{d_model}], got {tuple(basis.shape)}")
        self.register_buffer("basis", basis)
        self.scales = nn.Parameter(torch.ones(d_model, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_u = x @ self.basis
        x_u = x_u * self.scales
        return x_u @ self.basis.t()


class TwoHeadAttention(nn.Module):
    """2-head causal attention with fixed route bias and spectral Q/K/V/O."""

    def __init__(self, d_model: int, n_heads: int, route_bias: torch.Tensor, spectral_basis: torch.Tensor) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = SpectralLinearFixedU(d_model, spectral_basis)
        self.k_proj = SpectralLinearFixedU(d_model, spectral_basis)
        self.v_proj = SpectralLinearFixedU(d_model, spectral_basis)
        self.o_proj = SpectralLinearFixedU(d_model, spectral_basis)

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


class PhaseIOD4Adder(nn.Module):
    """4D phase representation model.

    Input channels for X tokens:
      0: d1_x
      1: d1_y
      2: d2_x
      3: d2_y

    Output interpretation at decode positions:
      [0:2] -> sum digit vector on unit circle
      [2:4] -> carry vector (2-class codebook)
    """

    def __init__(
        self,
        tok: Tokenizer,
        mlp_hidden: int = 2,
        mlp_bias_inner: bool = False,
        mlp_bias_outer: bool = False,
    ):
        super().__init__()
        self.tok = tok
        self.n_heads = 2
        self.d_model = 4
        self.max_len = 24
        self.prefix_len = 12
        self.mlp_hidden = mlp_hidden

        self.register_buffer("digit_basis", self._build_digit_basis())  # [10,2]
        self.register_buffer("carry_basis", self._build_carry_basis())  # [2,2]
        self.register_buffer("spectral_basis", self._build_dct_basis())  # [4,4]
        self.register_buffer("route_bias", self._build_route_bias())  # [2,24,24]

        self.attn = TwoHeadAttention(self.d_model, self.n_heads, self.route_bias, self.spectral_basis)
        self.fc1 = nn.Linear(self.d_model, self.mlp_hidden, bias=mlp_bias_inner)
        self.fc2 = nn.Linear(self.mlp_hidden, self.d_model, bias=mlp_bias_outer)

    @torch.no_grad()
    def _build_digit_basis(self) -> torch.Tensor:
        vecs = []
        for d in range(10):
            theta = 2.0 * math.pi * d / 10.0
            vecs.append([math.cos(theta), math.sin(theta)])
        return torch.tensor(vecs, dtype=torch.float32)

    @torch.no_grad()
    def _build_carry_basis(self) -> torch.Tensor:
        # Two orthogonal carry prototypes.
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    @torch.no_grad()
    def _build_dct_basis(self) -> torch.Tensor:
        n = self.d_model
        basis = torch.zeros((n, n), dtype=torch.float32)
        scale0 = math.sqrt(1.0 / n)
        scale = math.sqrt(2.0 / n)
        for k in range(n):
            alpha = scale0 if k == 0 else scale
            for i in range(n):
                basis[i, k] = alpha * math.cos(math.pi * (i + 0.5) * k / n)
        return basis

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
            rb[0, p, src_x] = 25.0
            rb[1, p, src_x] = 25.0
        return rb

    def _algorithmic_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        x = torch.zeros((bsz, seq_len, self.d_model), dtype=torch.float32, device=input_ids.device)

        # BOS behaves like X[0,0] for final carry column k=10.
        bos_mask = input_ids == self.tok.BOS
        if bos_mask.any():
            x[bos_mask, 0] = self.digit_basis[0, 0]
            x[bos_mask, 1] = self.digit_basis[0, 1]
            x[bos_mask, 2] = self.digit_basis[0, 0]
            x[bos_mask, 3] = self.digit_basis[0, 1]

        # X[d1,d2] -> [d1_vec, d2_vec]
        x_mask = (input_ids >= self.tok.X_BASE) & (input_ids < self.tok.X_BASE + 100)
        if x_mask.any():
            xv = input_ids[x_mask] - self.tok.X_BASE
            d1 = torch.div(xv, 10, rounding_mode="floor")
            d2 = xv % 10
            x[x_mask, 0] = self.digit_basis[d1, 0]
            x[x_mask, 1] = self.digit_basis[d1, 1]
            x[x_mask, 2] = self.digit_basis[d2, 0]
            x[x_mask, 3] = self.digit_basis[d2, 1]

        # Carry-coded tokens (C0 and Y[d,c]) use channels [2:4] as carry vector.
        c0_mask = input_ids == self.tok.C0
        if c0_mask.any():
            x[c0_mask, 2] = self.carry_basis[0, 0]
            x[c0_mask, 3] = self.carry_basis[0, 1]

        y0_mask = (input_ids >= self.tok.Y0_BASE) & (input_ids < self.tok.Y0_BASE + 10)
        if y0_mask.any():
            x[y0_mask, 2] = self.carry_basis[0, 0]
            x[y0_mask, 3] = self.carry_basis[0, 1]

        y1_mask = (input_ids >= self.tok.Y1_BASE) & (input_ids < self.tok.Y1_BASE + 10)
        if y1_mask.any():
            x[y1_mask, 2] = self.carry_basis[1, 0]
            x[y1_mask, 3] = self.carry_basis[1, 1]

        return x

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self._algorithmic_embed(input_ids)
        x = x + self.attn(x)
        x = x + self.fc2(F.relu(self.fc1(x)))
        return x

    def _decode_digit(self, vec2: torch.Tensor) -> torch.Tensor:
        return torch.argmax(vec2 @ self.digit_basis.t(), dim=-1)

    def _decode_carry(self, vec2: torch.Tensor) -> torch.Tensor:
        return torch.argmax(vec2 @ self.carry_basis.t(), dim=-1)

    def _algorithmic_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden.shape
        logits = torch.full((bsz, seq_len, self.tok.VOCAB_SIZE), -50.0, dtype=torch.float32, device=hidden.device)

        eos_pos = self.prefix_len - 1 + 11
        for p in range(seq_len):
            k = p - (self.prefix_len - 1)
            if 0 <= k <= 10:
                out_digit = self._decode_digit(hidden[:, p, 0:2])
                out_carry = self._decode_carry(hidden[:, p, 2:4]).clamp(0, 1)
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


class ScratchModel(PhaseIOD4Adder):
    def __init__(
        self,
        tok: Tokenizer,
        init_std: float,
        mlp_hidden: int = 2,
        mlp_bias_inner: bool = False,
        mlp_bias_outer: bool = False,
    ) -> None:
        super().__init__(
            tok,
            mlp_hidden=mlp_hidden,
            mlp_bias_inner=mlp_bias_inner,
            mlp_bias_outer=mlp_bias_outer,
        )
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


def compute_targets(a: int, b: int) -> Tuple[List[int], List[int]]:
    a_digits = [int(ch) for ch in reversed(f"{a:010d}")]
    b_digits = [int(ch) for ch in reversed(f"{b:010d}")]

    out_digits: List[int] = []
    out_carries: List[int] = []
    carry_in = 0
    for i in range(11):
        d1 = a_digits[i] if i < 10 else 0
        d2 = b_digits[i] if i < 10 else 0
        total = d1 + d2 + carry_in
        out_digit = total % 10
        carry_out = (total // 10) & 1
        out_digits.append(out_digit)
        out_carries.append(carry_out)
        carry_in = carry_out
    return out_digits, out_carries


def build_batch(tok: Tokenizer, batch_size: int, device: torch.device) -> Batch:
    rows: List[List[int]] = []
    gt_d_rows: List[List[int]] = []
    gt_c_rows: List[List[int]] = []

    for _ in range(batch_size):
        a = random.randrange(10**10)
        b = random.randrange(10**10)
        prefix = tok.encode_problem(a, b)
        out_d, out_c = compute_targets(a, b)

        y_tokens = [tok.y_id(out_d[i], out_c[i]) for i in range(11)]
        full_seq = prefix + y_tokens + [tok.EOS]
        rows.append(full_seq[:-1])
        gt_d_rows.append(out_d)
        gt_c_rows.append(out_c)

    return Batch(
        input_ids=torch.tensor(rows, dtype=torch.long, device=device),
        gt_out_digit=torch.tensor(gt_d_rows, dtype=torch.long, device=device),
        gt_out_carry=torch.tensor(gt_c_rows, dtype=torch.long, device=device),
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
    model = ScratchModel(
        tok,
        init_std=args.init_std,
        mlp_hidden=args.mlp_hidden,
        mlp_bias_inner=args.mlp_bias_inner,
        mlp_bias_outer=args.mlp_bias_outer,
    ).to(device)

    print(f"trainable_params={count_trainable_params(model)}")
    print(
        f"architecture: d_model=4 n_heads=2 mlp_hidden={args.mlp_hidden} "
        f"mlp_bias_inner={args.mlp_bias_inner} mlp_bias_outer={args.mlp_bias_outer} "
        f"spectral_qkvo=True"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
        h = model.forward_hidden(batch.input_ids)[:, start:end, :]  # [B,11,4]

        out_digit_logits = h[:, :, 0:2] @ model.digit_basis.t()
        out_carry_logits = h[:, :, 2:4] @ model.carry_basis.t()

        loss_d = F.cross_entropy(out_digit_logits.reshape(-1, 10), batch.gt_out_digit.reshape(-1))
        loss_c = F.cross_entropy(out_carry_logits.reshape(-1, 2), batch.gt_out_carry.reshape(-1))

        # Keep vectors near unit norm for stable nearest-basis decoding.
        sum_norm = torch.linalg.norm(h[:, :, 0:2], dim=-1)
        carry_norm = torch.linalg.norm(h[:, :, 2:4], dim=-1)
        loss_norm = 0.01 * (torch.mean((sum_norm - 1.0) ** 2) + torch.mean((carry_norm - 1.0) ** 2))

        loss = loss_d + loss_c + loss_norm
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
                f"(d={loss_d.item():.4f} c={loss_c.item():.4f} n={loss_norm.item():.4f}) "
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
    p = argparse.ArgumentParser(description="Train 4D phase IO spectral adder from scratch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--mlp-hidden", type=int, default=2)
    p.add_argument("--mlp-bias-inner", action="store_true")
    p.add_argument("--mlp-bias-outer", action="store_true")
    p.add_argument("--init-std", type=float, default=0.08)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-samples", type=int, default=256)
    p.add_argument("--final-eval-samples", type=int, default=10000)
    p.add_argument("--target-acc", type=float, default=1.1)
    p.add_argument("--checkpoint", type=str, default="checkpoints/phaseio_d4_spectral.pt")
    p.add_argument("--plot-path", type=str, default="checkpoints/phaseio_d4_spectral.png")
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    train(parse_args())
