import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2


class Classifier(nn.Module):
    def __init__(self, features_extractor, n_classes, weights=None):
        super().__init__()
        self.features_extractor = features_extractor
        self.n_classes = n_classes
        self.weights = weights

        self.classifier = None

        self.initialize()

    def initialize(self):
        self.define_classifier()

    def define_classifier(self):
        self.classifier = nn.Sequential(
            nn.Linear(self.features_extractor.latent_dim, self.n_classes)
        )
        if self.weights is not None:
            self.classifier.load_state_dict(torch.load(self.weights))

    def forward(self, x):
        embedding = self.features_extractor(x)
        y = self.classifier(embedding)
        return y

    def save(self):
        return self.classifier.state_dict()
