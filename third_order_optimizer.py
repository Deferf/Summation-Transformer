from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch


@dataclass
class ThirdOrderTRConfig:
    probes: int = 2
    sigma: float = 1.0
    sigma_min: float = 1e-4
    sigma_max: float = 1e3
    max_step: float = 0.05
    eta_accept: float = 0.1
    eta_good: float = 0.75
    grow_factor: float = 2.0
    shrink_factor: float = 0.7
    fallback_lr: float = 1e-3
    grad_tol: float = 1e-9


class ThirdOrderTrustRegionOptimizer:
    """Directional third-order trust-region optimizer.

    This optimizer uses random normalized probe directions and autograd-based
    directional derivatives (1st/2nd/3rd) to build a 1D cubic-regularized model.
    """

    def __init__(self, params: Sequence[torch.nn.Parameter], config: ThirdOrderTRConfig) -> None:
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("No trainable parameters provided")
        self.config = config

    @staticmethod
    def _fill_none(grads: Sequence[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for g, p in zip(grads, params):
            if g is None:
                out.append(torch.zeros_like(p))
            else:
                out.append(g)
        return out

    @staticmethod
    def _dot(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), device=xs[0].device)
        for x, y in zip(xs, ys):
            total = total + torch.sum(x * y)
        return total

    @staticmethod
    def _norm_sq(xs: Sequence[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), device=xs[0].device)
        for x in xs:
            total = total + torch.sum(x * x)
        return total

    def _random_unit_direction(self) -> List[torch.Tensor]:
        direction = [torch.randn_like(p) for p in self.params]
        norm = torch.sqrt(self._norm_sq(direction)) + 1e-12
        return [d / norm for d in direction]

    def _predict_decrease(self, g1: float, h2: float, t3: float, alpha: float) -> float:
        sigma = self.config.sigma
        model_delta = g1 * alpha + 0.5 * h2 * alpha * alpha
        model_delta += (t3 * alpha * alpha * alpha) / 6.0
        model_delta += (sigma * alpha * alpha * alpha * alpha) / 24.0
        return -model_delta

    def _choose_alpha(self, g1: float, h2: float, t3: float) -> Tuple[float, float]:
        denom = abs(h2) + 1e-8
        base = -g1 / denom
        max_step = self.config.max_step
        base = max(-max_step, min(max_step, base))

        candidates = [
            0.0,
            base,
            0.5 * base,
            1.5 * base,
            -base,
            max_step,
            -max_step,
            0.5 * max_step,
            -0.5 * max_step,
        ]

        best_alpha = 0.0
        best_pred = 0.0
        for alpha in candidates:
            pred = self._predict_decrease(g1, h2, t3, alpha)
            if math.isfinite(pred) and pred > best_pred:
                best_pred = pred
                best_alpha = alpha
        return best_alpha, best_pred

    @torch.no_grad()
    def _apply_step(self, direction: Sequence[torch.Tensor], alpha: float) -> None:
        for p, d in zip(self.params, direction):
            p.add_(d, alpha=alpha)

    @torch.no_grad()
    def _apply_gradient_fallback(self, grads: Sequence[torch.Tensor]) -> None:
        lr = self.config.fallback_lr
        if lr <= 0.0:
            return
        for p, g in zip(self.params, grads):
            p.add_(g, alpha=-lr)

    @torch.no_grad()
    def _snapshot(self) -> List[torch.Tensor]:
        return [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def _restore(self, snapshot: Sequence[torch.Tensor]) -> None:
        for p, old in zip(self.params, snapshot):
            p.copy_(old)

    def _update_sigma(self, accepted: bool, rho: float) -> None:
        if accepted:
            if rho > self.config.eta_good:
                self.config.sigma *= self.config.shrink_factor
            elif rho < self.config.eta_accept:
                self.config.sigma *= self.config.grow_factor
        else:
            self.config.sigma *= self.config.grow_factor
        self.config.sigma = float(max(self.config.sigma_min, min(self.config.sigma_max, self.config.sigma)))

    def step(self, closure: Callable[[], torch.Tensor]) -> Dict[str, float]:
        with torch.enable_grad():
            base_loss = closure()
            grads_raw = torch.autograd.grad(
                base_loss,
                self.params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            grads = self._fill_none(grads_raw, self.params)
            grad_norm = torch.sqrt(self._norm_sq([g.detach() for g in grads])).item()

            if grad_norm < self.config.grad_tol:
                return {
                    "loss": float(base_loss.detach().item()),
                    "trial_loss": float(base_loss.detach().item()),
                    "predicted_decrease": 0.0,
                    "actual_decrease": 0.0,
                    "rho": 0.0,
                    "accepted": 0.0,
                    "sigma": float(self.config.sigma),
                    "grad_norm": grad_norm,
                }

            best = {
                "direction": None,
                "alpha": 0.0,
                "pred": 0.0,
                "g1": 0.0,
                "h2": 0.0,
                "t3": 0.0,
            }

            for _ in range(self.config.probes):
                direction = self._random_unit_direction()
                g1 = self._dot(grads, direction)

                hv_raw = torch.autograd.grad(
                    g1,
                    self.params,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )
                hv = self._fill_none(hv_raw, self.params)
                h2 = self._dot(hv, direction)

                t_raw = torch.autograd.grad(
                    h2,
                    self.params,
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )
                t = self._fill_none(t_raw, self.params)
                t3 = self._dot(t, direction)

                g1_val = float(g1.detach().item())
                h2_val = float(h2.detach().item())
                t3_val = float(t3.detach().item())
                alpha, pred = self._choose_alpha(g1_val, h2_val, t3_val)

                if pred > best["pred"]:
                    best = {
                        "direction": [d.detach() for d in direction],
                        "alpha": alpha,
                        "pred": pred,
                        "g1": g1_val,
                        "h2": h2_val,
                        "t3": t3_val,
                    }

        snapshot = self._snapshot()
        accepted = False
        trial_loss_val = float(base_loss.detach().item())
        actual_decrease = 0.0
        rho = 0.0

        if best["direction"] is not None and best["pred"] > 0.0:
            self._apply_step(best["direction"], best["alpha"])
            with torch.no_grad():
                trial_loss = closure()
                trial_loss_val = float(trial_loss.item())
            base_loss_val = float(base_loss.detach().item())
            actual_decrease = base_loss_val - trial_loss_val
            rho = actual_decrease / (best["pred"] + 1e-12)

            accepted = actual_decrease > 0.0 and rho >= self.config.eta_accept
            if not accepted:
                self._restore(snapshot)

        if not accepted and self.config.fallback_lr > 0.0:
            detached_grads = [g.detach() for g in grads]
            self._apply_gradient_fallback(detached_grads)
            with torch.no_grad():
                fallback_loss = closure()
                fallback_loss_val = float(fallback_loss.item())
            if fallback_loss_val > float(base_loss.detach().item()):
                self._restore(snapshot)
                trial_loss_val = float(base_loss.detach().item())
            else:
                trial_loss_val = fallback_loss_val
                actual_decrease = float(base_loss.detach().item()) - trial_loss_val

        self._update_sigma(accepted, rho)

        return {
            "loss": float(base_loss.detach().item()),
            "trial_loss": trial_loss_val,
            "predicted_decrease": float(best["pred"]),
            "actual_decrease": float(actual_decrease),
            "rho": float(rho),
            "accepted": 1.0 if accepted else 0.0,
            "sigma": float(self.config.sigma),
            "grad_norm": float(grad_norm),
        }
