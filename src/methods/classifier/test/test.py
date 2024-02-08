import yaml

import torch
from torchvision.transforms import *

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.methods.classifier.model.features_extractor import FeaturesExtractor
from src.methods.classifier.model.classifier import Classifier


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
        self.predictions = []

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
        self.train_dataset = dataset.get_subset(
            split="val",
            transform=self.data_augmentation,
        )

    def define_dataloader(self):
        self.train_dataloader = get_train_loader(
            "standard",
            dataset=self.train_dataset,
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
        total = 0
        correct = 0
        with torch.no_grad():
            for iteration, labeled_batch in enumerate(zip(self.train_dataloader)):
                images, y, metadata = labeled_batch[0]
                images = images.to(self.device).to(torch.float32)
                labels = metadata[:, 0].to(self.device)

                predictions = self.model(images)
                _, predictions = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum()

                accuracy = torch.true_divide(correct, total).item()
                print("Accuracy: {:.3f} %".format(accuracy*100))

                self.labels += labels.cpu()
                self.predictions += predictions.cpu()
            self.confusion_matrix()

    def confusion_matrix(self):
        cell_lines = ["HEPG2", "HUVEC", "RPE", "U2OS"]
        cm = confusion_matrix(self.labels, self.predictions)
        display_cm = ConfusionMatrixDisplay(cm, display_labels=cell_lines)
        display_cm.plot(cmap="YlOrBr")
        plt.show()
