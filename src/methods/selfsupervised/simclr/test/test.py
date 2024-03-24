import yaml

import torch
from torchvision.transforms import *

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader

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

        self.labels = []
        self.embedding = []

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
        projection_head = ProjectionHead(
            latent_dim=self.config["model"]["project_head"]["latent_dim"],
            projection_dim=self.config["model"]["project_head"]["projection_dim"],
        )
        self.model = SimCLR(features_extractor, projection_head)
        self.model.to(self.device)
        self.model.eval()

    def test(self):
        cell_lines = ["HEPG2", "HUVEC", "RPE", "U2OS"]
        '''expriments = [
            "HEPG2-01", "HEPG2-02", "HEPG2-03", "HEPG2-04","HEPG2-05", "HEPG2-06", "HEPG2-07",
            "HEPG2-08", "HEPG2-09", "HEPG2-10", "HEPG2-11",
            "HUVEC-01", "HUVEC-02", "HUVEC-03", "HUVEC-04", "HUVEC-05", "HUVEC-06", "HUVEC-07",
            "HUVEC-08", "HUVEC-09", "HUVEC-10", "HUVEC-11", "HUVEC-12", "HUVEC-13", "HUVEC-14",
            "HUVEC-15", "HUVEC-16", "HUVEC-17", "HUVEC-18", "HUVEC-19", "HUVEC-20", "HUVEC-21",
            "HUVEC-22", "HUVEC-23", "HUVEC-24",
            "RPE-01", "RPE-02", "RPE-03", "RPE-04", "RPE-05", "RPE-06", "RPE-07", "RPE-08", "RPE-09", "RPE-10", "RPE-11",
            "U2OS-01", "U2OS-02", "U2OS-03", "U2OS-04", "U2OS-05"]'''
        tsne_visualization = TSNEVisualization(cell_lines)
        with torch.no_grad():
            for iteration, labeled_batch in enumerate(zip(self.eval_dataloader)):
                images, y, metadata = labeled_batch[0]
                image = images.to(self.device)
                embedding = self.model.encode(image)
                embedding = embedding.cpu().data
                labels = metadata[:, 0] #metadata[:, 1]
                self.labels.append(labels)
                self.embedding.append(embedding)
        self.labels = torch.cat(self.labels, dim=0)
        self.embedding = torch.cat(self.embedding, dim=0)
        tsne_visualization.visualize(self.embedding, self.labels)
