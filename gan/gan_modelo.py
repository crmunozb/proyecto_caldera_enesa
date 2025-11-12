# gan_modelo.py
"""
Definición de Generator y Discriminator para datos tabulares.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=32, data_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, data_dim),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, data_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),  # sin Sigmoid
        )

    def forward(self, x):
        return self.net(x)  # devuelve logits
