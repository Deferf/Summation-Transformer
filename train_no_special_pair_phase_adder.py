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


def build_digit_basis() -> torch.Tensor:
    vecs = []
    for d in range(10):
        theta = 2.0 * math.pi * d / 10.0
        vecs.append([math.cos(theta), math.sin(theta)])
    return torch.tensor(vecs, dtype=torch.float32)


@dataclass(frozen=True)
class PairTokenizer:
    """No-special-token tokenizer with pair tokens only.

    Each token represents one LSD-first digit column: X[d1,d2].
    token_id = 10*d1 + d2, range [0,99].
    """

    VOCAB_SIZE: int = 100

    def pair_id(self, d1: int, d2: int) -> int:
        if not (0 <= d1 <= 9 and 0 <= d2 <= 9):
            raise ValueError(f"Invalid digits: ({d1}, {d2})")
        return 10 * d1 + d2

    def id_to_pair(self, tok: int) -> Tuple[int, int]:
        if not (0 <= tok < self.VOCAB_SIZE):
            raise ValueError(f"Invalid pair token: {tok}")
        return tok // 10, tok % 10

    def encode_pairs(self, a: int, b: int, n_digits: int) -> List[int]:
        if not (0 <= a < 10**n_digits and 0 <= b < 10**n_digits):
            raise ValueError(f"a,b must be in [0,10^{n_digits})")
        a_digits = [int(ch) for ch in reversed(f"{a:0{n_digits}d}")]
        b_digits = [int(ch) for ch in reversed(f"{b:0{n_digits}d}")]
        return [self.pair_id(a_digits[i], b_digits[i]) for i in range(n_digits)]

    def decode_sum(self, digits_lsd: List[int], carries: List[int]) -> str:
        if not digits_lsd:
            return ""
        if len(digits_lsd) != len(carries):
            raise ValueError("digits_lsd and carries must have same length")
        final_carry = carries[-1]
        msd = "".join(str(d) for d in reversed(digits_lsd))
        return f"{final_carry}{msd}"


@dataclass
class Batch:
    input_ids: torch.Tensor
    gt_out_digit: torch.Tensor
    gt_out_carry: torch.Tensor


class SpectralLinearFixedU(nn.Module):
    """Square linear map with fixed orthogonal basis and trainable diagonal."""

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
    """Causal 2-head attention with fixed routing bias.

    Head 0 is biased to self (current column values).
    Head 1 is biased to previous token (carry routing).
    """

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


class NoSpecialPairPhaseAdder(nn.Module):
    """No-special-token phase adder.

    Input token embedding: X[d1,d2] -> [v(d1), v(d2)] in 4D.
    Output interpretation per position:
      hidden[0:2] ~ v(sum_digit)
      hidden[2:4] ~ v(carry_out), where carry_out in {0,1}
    """

    def __init__(
        self,
        tok: PairTokenizer,
        n_digits: int = 3,
        d_model: int = 4,
        n_heads: int = 2,
        mlp_hidden: int = 2,
        mlp_bias_inner: bool = False,
        mlp_bias_outer: bool = False,
    ) -> None:
        super().__init__()
        if d_model != 4:
            raise ValueError("This phase IO design expects d_model=4")
        if n_heads != 2:
            raise ValueError("This design expects n_heads=2")
        self.tok = tok
        self.n_digits = n_digits
        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_hidden = mlp_hidden

        digit_basis = build_digit_basis()
        carry_basis = torch.stack([digit_basis[0], digit_basis[1]], dim=0)
        self.register_buffer("digit_basis", digit_basis)  # [10,2]
        self.register_buffer("carry_basis", carry_basis)  # [2,2] == [v(0), v(1)]
        self.register_buffer("spectral_basis", self._build_dct_basis())  # [4,4]
        self.register_buffer("route_bias", self._build_route_bias())  # [2,n_digits,n_digits]

        self.attn = TwoHeadAttention(self.d_model, self.n_heads, self.route_bias, self.spectral_basis)
        self.fc1 = nn.Linear(self.d_model, self.mlp_hidden, bias=mlp_bias_inner)
        self.fc2 = nn.Linear(self.mlp_hidden, self.d_model, bias=mlp_bias_outer)

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
        rb = torch.full((self.n_heads, self.n_digits, self.n_digits), -25.0)
        for p in range(self.n_digits):
            # Head 0: self content (digit pair at same column).
            rb[0, p, p] = 25.0
            # Head 1: previous column carry route (or self for p=0).
            src_prev = p - 1 if p > 0 else p
            rb[1, p, src_prev] = 25.0
        return rb

    def _algorithmic_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.n_digits:
            raise ValueError(f"seq_len={seq_len} exceeds n_digits={self.n_digits}")
        x = torch.zeros((bsz, seq_len, self.d_model), dtype=torch.float32, device=input_ids.device)

        d1 = torch.div(input_ids, 10, rounding_mode="floor")
        d2 = input_ids % 10
        x[:, :, 0] = self.digit_basis[d1, 0]
        x[:, :, 1] = self.digit_basis[d1, 1]
        x[:, :, 2] = self.digit_basis[d2, 0]
        x[:, :, 3] = self.digit_basis[d2, 1]
        return x

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self._algorithmic_embed(input_ids)
        x = x + self.attn(x)
        x = x + self.fc2(F.relu(self.fc1(x)))
        return x

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.forward_hidden(input_ids)
        digit_logits = h[:, :, 0:2] @ self.digit_basis.t()  # [B,T,10]
        carry_logits = h[:, :, 2:4] @ self.carry_basis.t()  # [B,T,2]
        return digit_logits, carry_logits, h

    @torch.no_grad()
    def solve(self, a: int, b: int, device: torch.device | str = "cpu") -> str:
        dev = torch.device(device)
        ids = self.tok.encode_pairs(a, b, self.n_digits)
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        digit_logits, carry_logits, _ = self(x)
        pred_digits = torch.argmax(digit_logits[0], dim=-1).tolist()
        pred_carries = torch.argmax(carry_logits[0], dim=-1).tolist()
        return self.tok.decode_sum(pred_digits, pred_carries)


class ScratchModel(NoSpecialPairPhaseAdder):
    def __init__(
        self,
        tok: PairTokenizer,
        n_digits: int,
        init_std: float,
        d_model: int = 4,
        n_heads: int = 2,
        mlp_hidden: int = 2,
        mlp_bias_inner: bool = False,
        mlp_bias_outer: bool = False,
    ) -> None:
        super().__init__(
            tok=tok,
            n_digits=n_digits,
            d_model=d_model,
            n_heads=n_heads,
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
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
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
    fig.suptitle("No-Special Pair Phase Adder Metrics")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved plot: {out}")


def compute_targets(a: int, b: int, n_digits: int) -> Tuple[List[int], List[int]]:
    a_digits = [int(ch) for ch in reversed(f"{a:0{n_digits}d}")]
    b_digits = [int(ch) for ch in reversed(f"{b:0{n_digits}d}")]

    out_digits: List[int] = []
    out_carries: List[int] = []
    carry_in = 0
    for i in range(n_digits):
        total = a_digits[i] + b_digits[i] + carry_in
        out_digits.append(total % 10)
        carry_out = (total // 10) & 1
        out_carries.append(carry_out)
        carry_in = carry_out
    return out_digits, out_carries


def build_batch(tok: PairTokenizer, n_digits: int, batch_size: int, device: torch.device) -> Batch:
    rows: List[List[int]] = []
    gt_d_rows: List[List[int]] = []
    gt_c_rows: List[List[int]] = []

    for _ in range(batch_size):
        a = random.randrange(10**n_digits)
        b = random.randrange(10**n_digits)
        rows.append(tok.encode_pairs(a, b, n_digits))
        out_d, out_c = compute_targets(a, b, n_digits)
        gt_d_rows.append(out_d)
        gt_c_rows.append(out_c)

    return Batch(
        input_ids=torch.tensor(rows, dtype=torch.long, device=device),
        gt_out_digit=torch.tensor(gt_d_rows, dtype=torch.long, device=device),
        gt_out_carry=torch.tensor(gt_c_rows, dtype=torch.long, device=device),
    )


@torch.no_grad()
def eval_exact_acc(model: ScratchModel, n_digits: int, n_samples: int, device: torch.device) -> float:
    good = 0
    for _ in range(n_samples):
        a = random.randrange(10**n_digits)
        b = random.randrange(10**n_digits)
        pred = model.solve(a, b, device=device)
        exp = f"{a + b:0{n_digits + 1}d}"
        if pred == exp:
            good += 1
    return good / max(1, n_samples)


def print_example(args: argparse.Namespace) -> None:
    tok = PairTokenizer()
    basis = build_digit_basis()
    a = args.demo_a
    b = args.demo_b
    n_digits = args.n_digits

    ids = tok.encode_pairs(a, b, n_digits)
    out_d, out_c = compute_targets(a, b, n_digits)

    print(f"example: a={a:0{n_digits}d}, b={b:0{n_digits}d}, n_digits={n_digits}")
    print("input pair tokens (LSD->MSD):")
    for i, tid in enumerate(ids):
        d1, d2 = tok.id_to_pair(tid)
        v1 = basis[d1].tolist()
        v2 = basis[d2].tolist()
        print(
            f"  pos={i} token=X[{d1},{d2}] id={tid} "
            f"embed=[v({d1})={v1[0]:+.4f},{v1[1]:+.4f} ; v({d2})={v2[0]:+.4f},{v2[1]:+.4f}]"
        )

    print("target output vectors (LSD->MSD):")
    for i in range(n_digits):
        d = out_d[i]
        c = out_c[i]
        vd = basis[d].tolist()
        vc = basis[c].tolist()  # v(0) or v(1)
        print(
            f"  pos={i} target=[v(sum={d})={vd[0]:+.4f},{vd[1]:+.4f} ; "
            f"v(carry={c})={vc[0]:+.4f},{vc[1]:+.4f}]"
        )

    exp = f"{a + b:0{n_digits + 1}d}"
    print(f"expected full sum string: {exp}")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.show_example:
        print_example(args)

    tok = PairTokenizer()
    model = ScratchModel(
        tok=tok,
        n_digits=args.n_digits,
        init_std=args.init_std,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden=args.mlp_hidden,
        mlp_bias_inner=args.mlp_bias_inner,
        mlp_bias_outer=args.mlp_bias_outer,
    ).to(device)

    print(f"trainable_params={count_trainable_params(model)}")
    print(
        f"architecture: n_digits={args.n_digits} d_model={args.d_model} n_heads={args.n_heads} "
        f"mlp_hidden={args.mlp_hidden} mlp_bias_inner={args.mlp_bias_inner} "
        f"mlp_bias_outer={args.mlp_bias_outer} no_special_tokens=True"
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history_steps: List[int] = []
    history_losses: List[float] = []
    history_accs: List[float] = []
    last_step = 0
    last_loss = 0.0

    for step in range(1, args.steps + 1):
        last_step = step
        batch = build_batch(tok, args.n_digits, args.batch_size, device)
        digit_logits, carry_logits, h = model(batch.input_ids)

        loss_d = F.cross_entropy(digit_logits.reshape(-1, 10), batch.gt_out_digit.reshape(-1))
        loss_c = F.cross_entropy(carry_logits.reshape(-1, 2), batch.gt_out_carry.reshape(-1))

        sum_norm = torch.linalg.norm(h[:, :, 0:2], dim=-1)
        carry_norm = torch.linalg.norm(h[:, :, 2:4], dim=-1)
        loss_norm = 0.01 * (torch.mean((sum_norm - 1.0) ** 2) + torch.mean((carry_norm - 1.0) ** 2))
        loss = loss_d + loss_c + loss_norm
        last_loss = float(loss.item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == 1:
            acc = eval_exact_acc(model, args.n_digits, args.eval_samples, device)
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

    final_acc = eval_exact_acc(model, args.n_digits, args.final_eval_samples, device)
    print(f"final_exact_accuracy={final_acc:.4f} over {args.final_eval_samples} random sums")

    if not history_steps or history_steps[-1] != last_step:
        history_steps.append(last_step)
        history_losses.append(last_loss)
        history_accs.append(float(final_acc))

    save_training_plot(history_steps, history_losses, history_accs, args.plot_path)

    if args.checkpoint:
        out = Path(args.checkpoint)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "final_exact_accuracy": final_acc,
                "history_steps": history_steps,
                "history_losses": history_losses,
                "history_exact_acc": history_accs,
            },
            out,
        )
        print(f"saved checkpoint: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train no-special-token pair phase adder")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-digits", type=int, default=3)
    p.add_argument("--d-model", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--mlp-hidden", type=int, default=2)
    p.add_argument("--mlp-bias-inner", action="store_true")
    p.add_argument("--mlp-bias-outer", action="store_true")
    p.add_argument("--init-std", type=float, default=0.08)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-samples", type=int, default=512)
    p.add_argument("--final-eval-samples", type=int, default=5000)
    p.add_argument("--target-acc", type=float, default=1.1)
    p.add_argument("--checkpoint", type=str, default="checkpoints/no_special_pair_phase.pt")
    p.add_argument("--plot-path", type=str, default="checkpoints/no_special_pair_phase.png")
    p.add_argument("--show-example", action="store_true")
    p.add_argument("--demo-a", type=int, default=123)
    p.add_argument("--demo-b", type=int, default=456)
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    train(parse_args())
