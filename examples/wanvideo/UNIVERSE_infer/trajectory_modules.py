"""
Trajectory encoding and related helpers for conditioning diffusion models.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn 


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        point_dim: int = 3,
        velocity_dim: int = 2,
        hidden_dim: int = 256,
        output_dim: int = 4096,
    ) -> None:
        super().__init__()
        # Two-layer MLP for XY tokens (trajectory/history).
        self.traj_proj = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Velocity token (kept for backward compatibility).
        self.vel_proj = nn.Sequential(
            nn.Linear(velocity_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        trajectory: torch.Tensor,
        history_positions: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Future trajectory tokens always come last; pipeline code uses that
        # ordering to strip history/velocity prefixes before scheduler updates.
        traj_tokens = self.traj_proj(trajectory)

        tokens = []
        if history_positions is not None:
            # History tokens let DiT see recent ego motion without asking the
            # trajectory head to denoise those past points.
            hist_tokens = self.traj_proj(history_positions)
            tokens.append(hist_tokens)
        if velocity is not None:
            vel_token = self.vel_proj(velocity).unsqueeze(1)  # (B, 1, output_dim)
            tokens.append(vel_token)
        tokens.append(traj_tokens)

        return self.norm(torch.cat(tokens, dim=1))


class TrajectoryHead(nn.Module):
    def __init__(self, dim: int, out_dim: int = None) -> None:
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.norm.weight.dtype:
            x = x.to(self.norm.weight.dtype)
        return self.proj(self.norm(x))
