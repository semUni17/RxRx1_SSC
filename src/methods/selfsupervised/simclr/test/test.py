import yaml

import torch
from torchvision.transforms import *

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader

from src.methods.selfsupervised.simclr.utils.tsne_visualization import TSNEVisualization

from src.methods.classifier.model.features_extractor import FeaturesExtractor
from src.utils.dataaugmentation.self_standardization import SelfStandardization


class Test:
    def __init__(self, config_path: str):
        self.config_path = config_path

        self.device = None
        self.config = None
        self.transform = None
        self.eval_dataset = None
        self.eval_dataloader = None
        self.features_extractor = None
        self.tsne_visualization = None

        self.initialize()

    def initialize(self):
        self.define_config()
        self.define_transform()
        self.define_dataset()
        self.define_dataloader()
        self.define_model()

        self.tsne_visualization = TSNEVisualization(self.config["hyper_parameters"]["experiments"])

    def define_config(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)

        for k, v in self.config.items():
            print("{}: {}".format(k, v))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def define_transform(self):
        self.transform = Compose([
            ToTensor(),
            SelfStandardization()
        ])

    def define_dataset(self):
        dataset = get_dataset(
            dataset=self.config["dataset"]["name"],
            download=self.config["dataset"]["download"]
        )
        self.eval_dataset = dataset.get_subset(
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
        self.features_extractor = FeaturesExtractor(
            latent_dim=self.config["model"]["features_extractor"]["latent_dim"],
            weights=self.config["model"]["features_extractor"]["weights"]
        )
        self.features_extractor.to(self.device)
        self.features_extractor.eval()

    def test(self):
        with torch.no_grad():
            for iteration, labeled_batch in enumerate(zip(self.eval_dataloader)):
                images, y, metadata = labeled_batch[0]
                images = images.to(self.device).to(torch.float32)
                labels = metadata[:, 1].to(self.device)

                # mask = torch.logical_and(metadata[:, 1] >= 11, metadata[:, 1] <= 26)
                # images, y, metadata, labels = images[mask], y[mask], metadata[mask], labels[mask]

                embeddings = self.features_extractor(images)

                self.tsne_visualization.update(embeddings, labels)

        self.tsne_visualization.compute()
        self.tsne_visualization.plot_embeddings()
