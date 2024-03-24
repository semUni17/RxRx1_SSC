import numpy as np

import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, num_classes, average):
        self.num_classes = num_classes
        self.average = average

        self.device = None
        self.standard_accuracy = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.confusion = None
        self.confusion_matrix = None

        self.initialize()

    def initialize(self):
        self.define_config()
        self.define_metrics()

    def define_config(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def define_metrics(self):
        self.standard_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
        self.accuracy = MulticlassAccuracy(num_classes=self.num_classes, average=self.average)
        self.precision = MulticlassPrecision(num_classes=self.num_classes, average=self.average)
        self.recall = MulticlassRecall(num_classes=self.num_classes, average=self.average)
        self.f1_score = MulticlassF1Score(num_classes=self.num_classes, average=self.average)
        self.confusion = MulticlassConfusionMatrix(num_classes=self.num_classes)

        self.standard_accuracy.to(self.device)
        self.accuracy.to(self.device)
        self.precision.to(self.device)
        self.recall.to(self.device)
        self.f1_score.to(self.device)
        self.confusion.to(self.device)

    def update(self, labels, predictions):
        self.standard_accuracy.update(labels, predictions)
        self.accuracy.update(labels, predictions)
        self.precision.update(labels, predictions)
        self.recall.update(labels, predictions)
        self.f1_score.update(labels, predictions)
        self.confusion.update(labels, predictions)

    def compute(self):
        self.confusion_matrix = self.confusion.compute().cpu().numpy().astype(np.int32).T
        metrics = {
            "standard_accuracy": self.standard_accuracy.compute(),
            "accuracy": self.accuracy.compute(),
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
            "f1_score": self.f1_score.compute(),
            "confusion": self.confusion_matrix
        }
        return metrics

    def plot_confusion_matrix(self, class_names):
        display_cm = ConfusionMatrixDisplay(self.confusion_matrix, display_labels=class_names)
        display_cm.plot(cmap="YlOrBr")
        plt.title("Confusion Matrix")
        plt.show()
