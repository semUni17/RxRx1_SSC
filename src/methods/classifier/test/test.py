import numpy as np
import yaml

import torch
from torchvision.transforms import *

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader

from src.methods.classifier.model.features_extractor import FeaturesExtractor
from src.methods.classifier.model.classifier import Classifier
from src.utils.dataaugmentation.self_standardization import SelfStandardization

from src.utils.metrics.metrics import Metrics


class Test:
    def __init__(self, config_path: str):
        self.config_path = config_path

        self.device = None
        self.config = None
        self.transform = None
        self.eval_dataset = None
        self.eval_dataloader = None
        self.model = None
        self.metrics = None

        self.initialize()

    def initialize(self):
        self.define_config()
        self.define_transform()
        self.define_dataset()
        self.define_dataloader()
        self.define_model()

    def define_config(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)

        for k, v in self.config.items():
            print("{}: {}".format(k, v))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.metrics = Metrics(
            num_classes=self.config["model"]["classifier"]["n_classes"],
            average=self.config["hyper_parameters"]["average"]
        )

    def define_transform(self):
        self.transform = Compose([
            ToTensor(),
            SelfStandardization()
        ])

    def define_dataset(self):
        self.dataset = get_dataset(
            dataset=self.config["dataset"]["name"],
            download=self.config["dataset"]["download"],
        )
        self.eval_dataset = self.dataset.get_subset(
            split=self.config["dataset"]["split"],
            frac=self.config["dataset"]["fraction"],
            transform=self.transform,
        )

    def define_dataloader(self):
        self.eval_dataloader = get_eval_loader(
            "standard",
            dataset=self.eval_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            num_workers=self.config["hyper_parameters"]["num_workers"]
        )

    def define_model(self):
        features_extractor = FeaturesExtractor(
            latent_dim=self.config["model"]["features_extractor"]["latent_dim"],
            weights=self.config["model"]["features_extractor"]["weights"]
        )
        self.model = Classifier(
            features_extractor,
            self.config["model"]["classifier"]["n_classes"],
            self.config["model"]["classifier"]["weights"]
        )
        self.model.to(self.device)
        self.model.eval()

    def test(self):
        with torch.no_grad():
            for iteration, labeled_batch in enumerate(zip(self.eval_dataloader)):
                images, y, metadata = labeled_batch[0]
                images = images.to(self.device).to(torch.float32)
                labels = metadata[:, 0].to(self.device)

                predictions = self.model(images)
                _, predictions = torch.max(predictions.data, dim=1)

                self.metrics.update(labels, predictions)
                self.print_metrics(running=True)

            self.print_metrics(running=False)
            self.metrics.plot_confusion_matrix(class_names=self.config["hyper_parameters"]["class_names"])

    def print_metrics(self, running=True):
        metrics = self.metrics.compute()

        if running:
            print("{}: {:.3f} %".format("standard_accuracy", metrics["standard_accuracy"]*100))
        else:
            for k, v in metrics.items():
                if k != "confusion":
                    print("{}: {:.3f} %".format(k, v*100))
                else:
                    print("{}: {}".format(k, v))
