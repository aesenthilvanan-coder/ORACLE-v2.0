import torch
import numpy as np
from typing import Tuple


def linear_beta_schedule(
    n_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, n_timesteps)


def cosine_beta_schedule(
    n_timesteps: int,
    s: float = 0.008,
) -> torch.Tensor:
    steps = n_timesteps + 1
    x = torch.linspace(0, n_timesteps, steps)
    alpha_bars = torch.cos((x / n_timesteps + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    return betas.clamp(max=0.9999)


def compute_alpha_bars(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
    noise: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps"""
    if noise is None:
        noise = torch.randn_like(x0)
    ab = alpha_bars[t]
    while ab.dim() < x0.dim():
        ab = ab.unsqueeze(-1)
    x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
    return x_t, noise


def get_schedule_buffers(
    n_timesteps: int,
    schedule: str = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (betas, alphas, alpha_bars) for the chosen schedule."""
    if schedule == "cosine":
        betas = cosine_beta_schedule(n_timesteps)
    else:
        betas = linear_beta_schedule(n_timesteps, beta_start, beta_end)
    alphas, alpha_bars = compute_alpha_bars(betas)
    return betas, alphas, alpha_bars
