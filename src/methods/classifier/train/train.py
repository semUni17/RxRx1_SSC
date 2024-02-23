import yaml

import numpy as np
import cv2

import torch
from torch.nn import *
from torchvision.transforms import *
from torch.optim import *
from torch.optim.lr_scheduler import StepLR

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

from src.methods.classifier.model.features_extractor import FeaturesExtractor
from src.methods.classifier.model.classifier import Classifier
from src.utils.dataaugmentation.self_standardization import SelfStandardization


class Train:
    def __init__(self, config_path: str):
        self.config_path = config_path

        self.device = None
        self.config = None
        self.transform = None
        self.train_dataset = None
        self.train_dataloader = None
        self.model = None
        self.classifier = None
        self.criterion = None
        self.optimizer = None
        self.learning_rate = None

        self.initialize()

    def initialize(self):
        self.define_config()
        self.define_transform()
        self.define_dataset()
        self.define_dataloader()
        self.define_model()
        self.define_optimizer()

    def define_config(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)

        for k, v in self.config.items():
            print("{}: {}".format(k, v))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def define_transform(self):
        self.transform = Compose([
            ToTensor(),
            SelfStandardization(),
        ])

    def define_dataset(self):
        dataset = get_dataset(
            dataset=self.config["dataset"]["name"],
            download=self.config["dataset"]["download"]
        )
        self.train_dataset = dataset.get_subset(
            split=self.config["dataset"]["split"],
            frac=self.config["dataset"]["fraction"],
            transform=self.transform,
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
        self.model.train()

        for n, p in self.model.features_extractor.named_parameters():
            p.requires_grad = False

        '''print("ResNet")
        for n, p in self.model.features_extractor.named_parameters():
            print(p.requires_grad, end=" ")
        print()
        print("Classifier")
        for n, p in self.model.named_parameters():
            print(p.requires_grad, end=" ")
        print()'''

    def define_optimizer(self):
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.config["hyper_parameters"]["learning_rate"],
            weight_decay=self.config["hyper_parameters"]["weight_decay"]
        )
        self.learning_rate = StepLR(
            self.optimizer,
            step_size=self.config["hyper_parameters"]["step_size"],
            gamma=self.config["hyper_parameters"]["gamma"]
        )
        self.criterion = CrossEntropyLoss()

    def train(self):
        num_epochs = self.config["hyper_parameters"]["num_epochs"]
        for epoch in range(num_epochs):
            for iteration, labeled_batch in enumerate(zip(self.train_dataloader)):
                images, y, metadata = labeled_batch[0]
                images = images.to(self.device).to(torch.float32)
                labels = metadata[:, 0].to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(images)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()

                if (iteration % 10) == 0:
                    self.print_status(epoch, num_epochs, iteration, loss)

            self.save_checkpoint(epoch)
        self.save_checkpoint("final")

    @staticmethod
    def print_status(epoch, num_epochs, iteration, loss):
        print("Epoch {:03d}/{:03d} ({:06d}-th iteration) -> loss: {:.7f}".format(
            epoch+1,
            num_epochs,
            iteration,
            loss.item()
        ))

    def save_checkpoint(self, epoch):
        if (epoch % 5) == 0 or epoch == "final":
            name = self.config["model"]["features_extractor"]["save_path"]
            torch.save(self.model.save(), "{}_{}.{}".format(name, epoch, "pt"))
            print("Saved {}_{}.{}".format(name, epoch, "pt"))

