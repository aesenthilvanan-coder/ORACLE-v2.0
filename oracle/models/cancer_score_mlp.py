import torch
import torch.nn as nn
import numpy as np
from typing import List


class CancerScoreFunction(nn.Module):
    """Differentiable cancer score predictor — deep MLP with residual connections.

    Default config: hidden_dim=4096, n_hidden_layers=3 → ~170M parameters.
    Designed for 3B-scale target: hidden_dim=8192, n_hidden_layers=12 → ~1.1B params.
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 4096,
        n_hidden_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Deep hidden layers with pre-LN residual connections
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_hidden_layers)
        ])

        # Bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Skip connection from input
        self.skip = nn.Linear(n_genes, hidden_dim // 2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.hidden_layers:
            h = h + layer(h)
        h = self.bottleneck(h) + self.skip(x)
        return self.head(h).squeeze(-1)

    def gradient_wrt_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().requires_grad_(True)
        score = self.forward(x)
        score.sum().backward()
        return x.grad.detach()

    def batch_gradients(self, X: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        t = torch.tensor(X, dtype=torch.float32, device=device)
        grads = self.gradient_wrt_input(t)
        return grads.cpu().numpy()

    def score_numpy(self, X: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=device)
            return self.forward(t).cpu().numpy()
