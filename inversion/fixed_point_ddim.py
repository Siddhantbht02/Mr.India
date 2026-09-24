"""
Exact Fixed-Point DDIM Inversion Solver
Eliminates discretization trajectory drift via iterative fixed-point formulation
with FP32 arithmetic upcasting and bidirectional consistency validation.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from inversion.base_inversion import BaseInversion, compute_alpha


class FixedPointDDIMInverter(BaseInversion):
    """
    Iterative Fixed-Point DDIM Inversion & Generation Solver.
    
    Inversion update:
        x_{t+1}^(k+1) = sqrt(alpha_{t+1}) * x0_hat(x_t, t) 
                        + sqrt(1 - alpha_{t+1} - sigma_{t+1}^2) * eps_theta(x_{t+1}^(k), t+1) 
                        + sigma_{t+1} * noise
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 5,
        eta: float = 0.0,
        method: str = "fixed_point",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)
        self.tol = tol
        self.max_iter = max_iter
        self.eta = eta
        self.method = method

    def forward(
        self,
        x0: torch.Tensor,
        seq: List[int],
        model: nn.Module,
        betas: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward Inversion trajectory (x_0 -> x_T) using exact fixed-point iterations.
        
        Args:
            x0: Initial image / latent tensor, shape (B, C, H, W).
            seq: Subsampled sequence of timesteps [0, ..., T-1] in ascending order.
            model: Diffusion denoising model eps_theta(x, t).
            betas: Diffusion beta schedule tensor.
            
        Returns:
            Tuple of (xs, x0_preds):
                xs: List of latent states from x_0 to x_T.
                x0_preds: List of predicted x_0 at each step.
        """
        model_device = next(model.parameters()).device if list(model.parameters()) else self.device
        b = betas.to(model_device)
        
        # Upcast initial latents to FP32 to avoid rounding accumulation drift
        orig_dtype = x0.dtype
        x = x0.to(device=model_device, dtype=torch.float32)
        n = x.size(0)

        seq_clean = list(seq)
        seq_next = list(seq_clean)
        seq_curr = [-1] + list(seq_clean[:-1])

        xs = [x]
        x0_preds = []
        tol = kwargs.get("tol", self.tol)
        max_iter = kwargs.get("max_iter", self.max_iter)
        eta = kwargs.get("eta", self.eta)

        with torch.no_grad():
            for t_idx, next_t_idx in zip(seq_curr, seq_next):
                t = torch.full((n,), t_idx, device=model_device, dtype=torch.long)
                next_t = torch.full((n,), next_t_idx, device=model_device, dtype=torch.long)

                # Compute alpha at t and t+1 in FP32
                at = compute_alpha(b, t)
                at_next = compute_alpha(b, next_t)

                xt = xs[-1].to(torch.float32)

                # Evaluate model at current state xt, t
                # If t == -1 (initial image), we define at = 1.0, et is evaluated at t=0
                if t_idx < 0:
                    et = model(xt.to(orig_dtype), torch.zeros_like(next_t)).to(torch.float32)
                    x0_t = xt
                else:
                    et = model(xt.to(orig_dtype), t).to(torch.float32)
                    x0_t = (xt - et * (1.0 - at).sqrt()) / at.sqrt()

                x0_preds.append(x0_t)

                # Compute forward variance if stochastic DDIM is requested (default eta=0)
                if eta > 0:
                    sigma = eta * ((1.0 - at_next / at) * (1.0 - at) / (1.0 - at_next)).clamp(min=0.0).sqrt()
                else:
                    sigma = torch.zeros(1, device=model_device, dtype=torch.float32)

                c2 = (1.0 - at_next - sigma ** 2).clamp(min=0.0).sqrt()

                # k=0 initialization: standard naive DDIM forward estimate
                xt_next_k = at_next.sqrt() * x0_t + c2 * et

                method = kwargs.get("method", self.method)
                if method == "naive" or max_iter == 0:
                    xs.append(xt_next_k)
                    continue

                # Iterative Fixed-Point formulation:
                # x_{t+1}^(k+1) = sqrt(alpha_{t+1}) * x0_t + c2 * eps_theta(x_{t+1}^(k), t+1)
                best_xt_next = xt_next_k
                min_diff = float("inf")

                for k in range(max_iter):
                    et_next = model(xt_next_k.to(orig_dtype), next_t).to(torch.float32)
                    noise = torch.randn_like(xt) if sigma.item() > 0 else 0.0
                    xt_next_cand = at_next.sqrt() * x0_t + c2 * et_next + sigma * noise

                    diff = torch.norm(xt_next_cand - xt_next_k, p=2) / (xt.numel() ** 0.5)
                    diff_val = diff.item()

                    if diff_val < min_diff:
                        min_diff = diff_val
                        best_xt_next = xt_next_cand

                    if diff_val <= tol:
                        xt_next_k = xt_next_cand
                        break

                    xt_next_k = xt_next_cand
                else:
                    # If max iterations reached, apply fallback to best iterate found
                    xt_next_k = best_xt_next

                xs.append(xt_next_k)

        return xs, x0_preds

    def backward(
        self,
        xt: torch.Tensor,
        seq: List[int],
        model: nn.Module,
        betas: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Backward Generation trajectory (x_T -> x_0).
        
        Args:
            xt: Starting noise latent tensor at T, shape (B, C, H, W).
            seq: Subsampled sequence of timesteps [0, ..., T-1] in ascending order.
            model: Diffusion denoising model eps_theta(x, t).
            betas: Diffusion beta schedule tensor.
            
        Returns:
            Tuple of (xs, x0_preds):
                xs: List of generation states from x_T down to x_0.
                x0_preds: List of predicted x_0 at each step.
        """
        model_device = next(model.parameters()).device if list(model.parameters()) else self.device
        b = betas.to(model_device)
        orig_dtype = xt.dtype
        x = xt.to(device=model_device, dtype=torch.float32)
        n = x.size(0)

        seq_clean = list(seq)
        seq_prev = [-1] + list(seq_clean[:-1])

        xs = [x]
        x0_preds = []
        eta = kwargs.get("eta", self.eta)

        with torch.no_grad():
            for curr_t, prev_t in zip(reversed(seq_clean), reversed(seq_prev)):
                t = torch.full((n,), curr_t, device=model_device, dtype=torch.long)
                prev_t_tensor = torch.full((n,), prev_t, device=model_device, dtype=torch.long)

                at = compute_alpha(b, t)
                a_prev = compute_alpha(b, prev_t_tensor)

                curr_x = xs[-1].to(torch.float32)
                et = model(curr_x.to(orig_dtype), t).to(torch.float32)

                x0_t = (curr_x - et * (1.0 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t)

                if eta > 0 and prev_t >= 0:
                    c1 = eta * ((1.0 - at / a_prev) * (1.0 - a_prev) / (1.0 - at)).clamp(min=0.0).sqrt()
                else:
                    c1 = torch.zeros(1, device=model_device, dtype=torch.float32)

                c2 = (1.0 - a_prev - c1 ** 2).clamp(min=0.0).sqrt()
                noise = torch.randn_like(curr_x) if c1.item() > 0 else 0.0

                x_prev = a_prev.sqrt() * x0_t + c1 * noise + c2 * et
                xs.append(x_prev)

        return xs, x0_preds

    def verify_consistency(
        self,
        x0: torch.Tensor,
        seq: List[int],
        model: nn.Module,
        betas: torch.Tensor,
        tol: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Executes a bidirectional loop: x_0 -> x_T -> x_hat_0.
        Computes L2 reconstruction error, PSNR, and maximum coordinate drift.
        """
        tolerance = tol if tol is not None else self.tol
        
        # Forward inversion
        xs_fwd, _ = self.forward(x0, seq, model, betas, tol=tolerance)
        xT = xs_fwd[-1]

        # Backward generation
        xs_bwd, _ = self.backward(xT, seq, model, betas)
        x0_recon = xs_bwd[-1]

        diff = (x0 - x0_recon).abs()
        l2_err = torch.norm(x0 - x0_recon, p=2).item() / (x0.numel() ** 0.5)
        max_err = diff.max().item()
        mean_err = diff.mean().item()

        # Compute PSNR
        mse = torch.mean((x0 - x0_recon) ** 2).item()
        psnr = 20.0 * torch.log10(torch.tensor(2.0) / (torch.tensor(mse).sqrt() + 1e-12)).item()

        return {
            "l2_error": l2_err,
            "max_error": max_err,
            "mean_error": mean_err,
            "psnr": psnr,
            "steps": len(seq),
            "is_consistent": bool(l2_err <= 1e-4),
        }
