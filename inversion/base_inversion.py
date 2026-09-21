"""
Base Inversion Module
Defines the abstract interface for diffusion model forward inversion and backward sampling.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Computes cumulative product of (1 - beta) at given timesteps t.
    Supports 1D batch of timesteps, returning shape (batch, 1, 1, 1).
    Ensures FP32 precision during computation.
    """
    orig_device = beta.device
    beta_fp32 = beta.to(torch.float32)
    beta_with_zero = torch.cat([torch.zeros(1, device=orig_device, dtype=torch.float32), beta_fp32], dim=0)
    alphas = (1.0 - beta_with_zero).cumprod(dim=0)
    
    # t is 0-indexed; clamp to valid range
    t_clamped = torch.clamp(t + 1, 0, len(beta_with_zero) - 1).long()
    a = alphas.index_select(0, t_clamped).view(-1, 1, 1, 1)
    return a


class BaseInversion(ABC):
    """
    Abstract Base Class for diffusion inversion solvers.
    """

    def __init__(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

    @abstractmethod
    def forward(
        self,
        x0: torch.Tensor,
        seq: List[int],
        model: nn.Module,
        betas: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward trajectory (inversion): maps image/latent x_0 to inverted latent x_T.
        
        Args:
            x0: Initial image or latent tensor of shape (B, C, H, W).
            seq: Ordered list of diffusion timesteps [t_0, t_1, ..., t_T].
            model: Noise prediction network epsilon_theta(x, t).
            betas: Variance schedule tensor.
            
        Returns:
            Tuple of (trajectory_states [x_0, x_t1, ..., x_T], predicted_x0_list).
        """
        pass

    @abstractmethod
    def backward(
        self,
        xt: torch.Tensor,
        seq: List[int],
        model: nn.Module,
        betas: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Backward trajectory (generation): maps latent x_T to reconstructed image x_0.
        
        Args:
            xt: Starting latent tensor of shape (B, C, H, W).
            seq: Ordered list of diffusion timesteps [t_0, t_1, ..., t_T].
            model: Noise prediction network epsilon_theta(x, t).
            betas: Variance schedule tensor.
            
        Returns:
            Tuple of (trajectory_states [x_T, ..., x_0], predicted_x0_list).
        """
        pass
