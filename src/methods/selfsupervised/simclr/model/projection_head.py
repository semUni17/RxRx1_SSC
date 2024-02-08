import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2


class ProjectionHead(nn.Module):
    def __init__(self, latent_dim, projection_dim, weights=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim
        self.weights = weights

        self.projection_head = None

        self.initialize()

    def initialize(self):
        self.define_projection_head()

    def define_projection_head(self):
        self.projection_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.projection_dim, bias=False),
            nn.BatchNorm1d(self.projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.projection_dim, self.projection_dim, bias=False),
            nn.BatchNorm1d(self.projection_dim)
        )

    def forward(self, x):
        y = self.projection_head(x)
        y = F.normalize(y)
        return y

    def save(self):
        return self.projection_head.state_dict()
