from __future__ import annotations

import argparse
import random
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from addition_data import AdditionTokenizer, build_eval_set, build_train_batch
from model import DecoderOnlyTransformer, ModelConfig
from third_order_optimizer import ThirdOrderTRConfig, ThirdOrderTrustRegionOptimizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss(model: DecoderOnlyTransformer, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))


@torch.no_grad()
def eval_teacher_forced_loss(
    model: DecoderOnlyTransformer,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    losses: List[float] = []
    for start in range(0, x_eval.size(0), batch_size):
        x = x_eval[start : start + batch_size]
        y = y_eval[start : start + batch_size]
        loss = compute_loss(model, x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def generate_sum_text(
    model: DecoderOnlyTransformer,
    tokenizer: AdditionTokenizer,
    a: int,
    b: int,
    device: torch.device,
) -> str:
    prompt = f"^{a:010d}+{b:010d}="
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output = model.generate(input_ids, max_new_tokens=12)
    return tokenizer.decode(output[0].tolist())


@torch.no_grad()
def eval_exact_match_accuracy(
    model: DecoderOnlyTransformer,
    tokenizer: AdditionTokenizer,
    pairs: List[Tuple[int, int]],
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    for a, b in pairs:
        text = generate_sum_text(model, tokenizer, a, b, device)
        expected = f"{(a + b):011d}"

        eq_idx = text.find("=")
        dollar_idx = text.find("$", eq_idx + 1)
        if eq_idx == -1:
            predicted = ""
        elif dollar_idx == -1:
            predicted = text[eq_idx + 1 :]
        else:
            predicted = text[eq_idx + 1 : dollar_idx]

        if predicted == expected:
            correct += 1
    model.train()
    return correct / max(1, len(pairs))


@torch.no_grad()
def perturb_model(model: DecoderOnlyTransformer, std: float) -> None:
    for p in model.parameters():
        if p.requires_grad:
            p.add_(torch.randn_like(p) * std)


def model_score(exact_acc: float, eval_loss: float) -> Tuple[float, float]:
    # Maximize exact accuracy first, then minimize teacher-forced loss.
    return exact_acc, -eval_loss


def run_training_phase(
    model: DecoderOnlyTransformer,
    tokenizer: AdditionTokenizer,
    optimizer: ThirdOrderTrustRegionOptimizer,
    device: torch.device,
    steps: int,
    batch_size: int,
    log_interval: int,
) -> Dict[str, float]:
    latest_stats: Dict[str, float] = {}
    start_time = time.time()

    for step in range(1, steps + 1):
        x_train, y_train = build_train_batch(tokenizer, batch_size=batch_size, device=device)

        def closure() -> torch.Tensor:
            return compute_loss(model, x_train, y_train)

        stats = optimizer.step(closure)
        latest_stats = stats

        if step % log_interval == 0 or step == steps:
            elapsed = time.time() - start_time
            print(
                f"  step={step:5d}/{steps} "
                f"loss={stats['loss']:.4f} trial={stats['trial_loss']:.4f} "
                f"pred={stats['predicted_decrease']:.4e} actual={stats['actual_decrease']:.4e} "
                f"rho={stats['rho']:.3f} accept={int(stats['accepted'])} "
                f"sigma={stats['sigma']:.3e} grad_norm={stats['grad_norm']:.3e} "
                f"time={elapsed:.1f}s"
            )

    return latest_stats


def train_one_restart(
    restart_id: int,
    args: argparse.Namespace,
    tokenizer: AdditionTokenizer,
    eval_pairs: List[Tuple[int, int]],
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    device: torch.device,
) -> Tuple[DecoderOnlyTransformer, Dict[str, float]]:
    set_seed(args.seed + restart_id)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=tokenizer.model_block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
    )
    model = DecoderOnlyTransformer(cfg).to(device)

    opt_cfg = ThirdOrderTRConfig(
        probes=args.probes,
        sigma=args.sigma,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        max_step=args.max_step,
        eta_accept=args.eta_accept,
        eta_good=args.eta_good,
        grow_factor=args.grow_factor,
        shrink_factor=args.shrink_factor,
        fallback_lr=args.fallback_lr,
    )
    optimizer = ThirdOrderTrustRegionOptimizer(model.parameters(), opt_cfg)

    print(f"restart={restart_id} main phase")
    run_training_phase(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        steps=args.train_steps,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )

    best_state = deepcopy(model.state_dict())
    best_eval_loss = eval_teacher_forced_loss(model, x_eval, y_eval, batch_size=args.eval_batch_size)
    best_exact = eval_exact_match_accuracy(model, tokenizer, eval_pairs, device)
    print(f"restart={restart_id} eval loss={best_eval_loss:.4f} exact={best_exact:.4f}")

    for hop in range(1, args.basin_hops + 1):
        model.load_state_dict(best_state)
        perturb_model(model, std=args.basin_perturb_std)

        print(f"restart={restart_id} basin-hop={hop}")
        run_training_phase(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            steps=args.hop_steps,
            batch_size=args.batch_size,
            log_interval=args.log_interval,
        )

        eval_loss = eval_teacher_forced_loss(model, x_eval, y_eval, batch_size=args.eval_batch_size)
        exact = eval_exact_match_accuracy(model, tokenizer, eval_pairs, device)
        print(f"restart={restart_id} basin-hop={hop} eval loss={eval_loss:.4f} exact={exact:.4f}")

        if model_score(exact, eval_loss) > model_score(best_exact, best_eval_loss):
            best_state = deepcopy(model.state_dict())
            best_eval_loss = eval_loss
            best_exact = exact

    model.load_state_dict(best_state)
    metrics = {
        "eval_loss": best_eval_loss,
        "exact_accuracy": best_exact,
        "restart_id": float(restart_id),
    }
    return model, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train decoder-only transformer for 10-digit addition with a 3rd-order optimizer")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--mlp-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--train-steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--log-interval", type=int, default=25)

    p.add_argument("--basin-hops", type=int, default=1)
    p.add_argument("--hop-steps", type=int, default=80)
    p.add_argument("--basin-perturb-std", type=float, default=0.002)

    p.add_argument("--probes", type=int, default=2)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--sigma-min", type=float, default=1e-4)
    p.add_argument("--sigma-max", type=float, default=1e3)
    p.add_argument("--max-step", type=float, default=0.05)
    p.add_argument("--eta-accept", type=float, default=0.1)
    p.add_argument("--eta-good", type=float, default=0.75)
    p.add_argument("--grow-factor", type=float, default=2.0)
    p.add_argument("--shrink-factor", type=float, default=0.7)
    p.add_argument("--fallback-lr", type=float, default=1e-3)

    p.add_argument("--eval-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_addition_3rd_order.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AdditionTokenizer.build()
    x_eval, y_eval, eval_pairs = build_eval_set(tokenizer, size=args.eval_size, seed=args.seed + 10_000, device=device)

    best_model = None
    best_metrics = {
        "eval_loss": float("inf"),
        "exact_accuracy": -1.0,
        "restart_id": -1.0,
    }

    for restart_id in range(args.restarts):
        model, metrics = train_one_restart(
            restart_id=restart_id,
            args=args,
            tokenizer=tokenizer,
            eval_pairs=eval_pairs,
            x_eval=x_eval,
            y_eval=y_eval,
            device=device,
        )

        if model_score(metrics["exact_accuracy"], metrics["eval_loss"]) > model_score(
            best_metrics["exact_accuracy"], best_metrics["eval_loss"]
        ):
            best_model = model
            best_metrics = metrics

    if best_model is None:
        raise RuntimeError("No model produced")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "model_config": asdict(best_model.config),
            "tokenizer_stoi": tokenizer.stoi,
            "best_metrics": best_metrics,
            "train_args": vars(args),
        },
        ckpt_path,
    )

    print("training complete")
    print(
        f"best restart={int(best_metrics['restart_id'])} "
        f"eval_loss={best_metrics['eval_loss']:.4f} "
        f"exact_accuracy={best_metrics['exact_accuracy']:.4f}"
    )
    print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
