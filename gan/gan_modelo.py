# gan_modelo.py
"""
Definición de Generator y Discriminator (critic) para datos tabulares.

- Generator: más capacidad + BatchNorm para estabilizar entrenamiento.
- Discriminator: actúa como CRITIC en WGAN-GP (sin Sigmoid, sin BCE).
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 32, data_dim: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, data_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch_size, latent_dim)
        return self.net(z)


class Discriminator(nn.Module):
    """
    En WGAN-GP este módulo actúa como CRITIC:
    - No tiene Sigmoid.
    - Devuelve un escalar real (logit) por muestra.
    """
    def __init__(self, data_dim: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, data_dim)
        return self.net(x)  # valores reales (no probabilidades)
