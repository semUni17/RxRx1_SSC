import yaml

import torch
from torchvision.transforms import *

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

from src.methods.selfsupervised.simclr.utils.tsne_visualization import TSNEVisualization

from src.methods.classifier.model.features_extractor import FeaturesExtractor
from src.methods.selfsupervised.simclr.model.projection_head import ProjectionHead
from src.methods.selfsupervised.simclr.model.simclr import SimCLR


class Test:
    def __init__(self, config_path: str):
        self.config_path = config_path

        self.device = None
        self.config = None
        self.data_augmentation = None
        self.eval_dataset = None
        self.eval_dataloader = None
        self.model = None

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

    def define_transform(self):
        self.data_augmentation = Compose([
            ToTensor()
        ])

    def define_dataset(self):
        dataset = get_dataset(
            dataset=self.config["dataset"]["name"],
            download=self.config["dataset"]["download"]
        )
        self.eval_dataset = dataset.get_subset(
            split="val",
            transform=self.data_augmentation,
        )

    def define_dataloader(self):
        self.eval_dataloader = get_train_loader(
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
        projection_head = ProjectionHead(
            latent_dim=self.config["model"]["project_head"]["latent_dim"],
            projection_dim=self.config["model"]["project_head"]["projection_dim"],
        )
        self.model = SimCLR(features_extractor, projection_head)
        self.model.to(self.device)
        self.model.eval()

    def test(self):
        cell_lines = ["HEPG2", "HUVEC", "RPE", "U2OS"]
        tsne_visualization = TSNEVisualization(cell_lines)
        with torch.no_grad():
            for iteration, labeled_batch in enumerate(zip(self.eval_dataloader)):
                images, y, metadata = labeled_batch[0]
                image = images.to(self.device)
                embedding = self.model.encode(image)
                embedding = embedding.cpu().data
                labels = metadata[:, 0]
                tsne_visualization.visualize(embedding, labels)
