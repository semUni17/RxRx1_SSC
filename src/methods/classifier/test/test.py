import numpy as np
import yaml

import torch
from torchvision.transforms import *
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.methods.classifier.model.features_extractor import FeaturesExtractor
from src.methods.classifier.model.classifier import Classifier
from src.utils.dataaugmentation.self_standardization import SelfStandardization


class Test:
    def __init__(self, config_path: str):
        self.config_path = config_path

        self.device = None
        self.config = None
        self.transform = None
        self.eval_dataset = None
        self.eval_dataloader = None
        self.model = None

        self.y = []
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
        total = 0
        correct = 0
        l = []
        self.metadata = []

        num_classes = self.config["model"]["classifier"]["n_classes"]
        confusion = MulticlassConfusionMatrix(num_classes=num_classes)
        accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")
        precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        recall = MulticlassRecall(num_classes=num_classes, average="macro")
        f1_score = MulticlassF1Score(num_classes=num_classes, average="macro")

        with torch.no_grad():
            for iteration, labeled_batch in enumerate(zip(self.eval_dataloader)):
                images, y, metadata = labeled_batch[0]
                images = images.to(self.device).to(torch.float32)
                labels = metadata[:, 0].to(self.device)

                l += metadata[:, 1].tolist()

                predictions = self.model(images)
                _, predictions = torch.max(predictions.data, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum()

                manual_accuracy = torch.true_divide(correct, total).item()
                print("Manual accuracy: {:.3f} %".format(manual_accuracy*100))

                self.labels += labels.cpu()
                self.predictions += predictions.cpu()
                self.metadata += metadata

                accuracy.update(labels, predictions)
                precision.update(labels, predictions)
                recall.update(labels, predictions)
                f1_score.update(labels, predictions)
                confusion.update(labels, predictions)

            a = accuracy.compute()
            p = precision.compute()
            r = recall.compute()
            f1 = f1_score.compute()
            print("Accuracy: {:.3f} %".format(a*100))
            print("Precision: {:.3f} %".format(p*100))
            print("Recall: {:.3f} %".format(r*100))
            print("F1 score: {:.3f} %".format(f1*100))

            print(len(l), len(set(l)), set(l))

            self.confusion_matrix(confusion.compute())

    def confusion_matrix(self, confusion):
        cell_lines = ["HEPG2", "HUVEC", "RPE", "U2OS"]
        cm = confusion_matrix(self.labels, self.predictions)
        print(cm)
        cm = confusion.cpu().numpy().astype(np.int32).T
        display_cm = ConfusionMatrixDisplay(cm, display_labels=cell_lines)
        display_cm.plot(cmap="YlOrBr")
        plt.show()
