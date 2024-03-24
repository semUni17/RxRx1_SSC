import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import numpy as np
import cv2


class FeaturesExtractor(nn.Module):
    def __init__(self, latent_dim, weights="default"):
        super().__init__()
        self.latent_dim = latent_dim
        self.weights = weights

        self.features_extractor = None

        self.initialize()

    def initialize(self):
        self.define_features_extractor()

    def define_features_extractor(self):
        if self.weights == "default":
            print("Default pre-trained ResNet50")
            self.features_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.features_extractor.fc = nn.Linear(self.features_extractor.fc.in_features, self.latent_dim, bias=True)
        else:
            print("Custom pre-trained ResNet50")
            self.features_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.features_extractor.fc = nn.Linear(self.features_extractor.fc.in_features, self.latent_dim, bias=True)
            try:
                self.features_extractor.load_state_dict(torch.load(self.weights))
            except:
                pre_trained_features_extractor_dict = torch.load(self.weights)
                features_extractor_dict = self.features_extractor.state_dict()
                # {k.split(".", 1)[1]: v for k, v in pretrained_dict.items() if k.split(".", 1)[1] in model_dict}
                new_pre_trained_features_extractor_dict = {}
                for k, v in pre_trained_features_extractor_dict.items():
                    k = k.split(".", 1)[1]
                    if k in features_extractor_dict:
                        new_pre_trained_features_extractor_dict[k] = v
                features_extractor_dict.update(new_pre_trained_features_extractor_dict)
                self.features_extractor.load_state_dict(features_extractor_dict)

    def forward(self, x):
        y = self.features_extractor(x)
        return y

    def save(self):
        return self.features_extractor.state_dict()
