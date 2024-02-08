import yaml

import torch
from torchvision.transforms import *
from torch.optim import *
from torch.cuda.amp import GradScaler

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

from src.methods.selfsupervised.simclr.dataaugmentation.data_augmentation import DataAugmentation
from src.methods.classifier.model.features_extractor import FeaturesExtractor
from src.methods.selfsupervised.simclr.model.projection_head import ProjectionHead
from src.methods.selfsupervised.simclr.model.simclr import SimCLR


class Train:
    def __init__(self, config_path: str):
        self.config_path = config_path

        self.config = None
        self.device = None
        self.data_augmentation = None
        self.train_dataset = None
        self.train_dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

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
        scaler = self.config["data_augmentation"]["scaler"]
        size = self.config["data_augmentation"]["size"]
        self.data_augmentation = Compose([
            Resize(size=size),
            RandomResizedCrop(size=size),
            RandomHorizontalFlip(),
            RandomApply([transforms.ColorJitter(0.8*scaler, 0.8*scaler, 0.8*scaler, 0.2*scaler)], p=0.8),
            RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=self.config["data_augmentation"]["kernel_size"]),
            ToTensor()
        ])

    def define_dataset(self):
        dataset = get_dataset(
            dataset=self.config["dataset"]["name"],
            download=self.config["dataset"]["download"]
        )
        self.train_dataset = dataset.get_subset(
            split="train",
            transform=DataAugmentation(self.data_augmentation),
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
        projection_head = ProjectionHead(
            latent_dim=self.config["model"]["project_head"]["latent_dim"],
            projection_dim=self.config["model"]["project_head"]["projection_dim"],
        )
        self.model = SimCLR(features_extractor, projection_head)
        self.model.to(self.device)
        self.model.train()

    def define_optimizer(self):
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.config["hyper_parameters"]["learning_rate"],
            weight_decay=self.config["hyper_parameters"]["weight_decay"]
        )
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_dataloader),
            eta_min=self.config["hyper_parameters"]["eta_min"],
            last_epoch=self.config["hyper_parameters"]["last_epoch"]
        )
        self.scaler = GradScaler()

    def train(self):
        num_epochs = self.config["hyper_parameters"]["num_epochs"]
        for epoch in range(num_epochs):
            for iteration, labeled_batch in enumerate(zip(self.train_dataloader)):
                images, y, metadata = labeled_batch[0]
                for i in range(len(images)):
                    images[i] = images[i].to(self.device).to(torch.float32)

                self.optimizer.zero_grad()
                loss = self.model(images)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.print_status(epoch, num_epochs, iteration, loss)

            self.save_model()
        self.save_model()

    @staticmethod
    def print_status(epoch, num_epochs, iteration, loss):
        if (iteration % 10) == 0:
            print("Epoch {:03d}/{:03d} ({:06d}-th iteration) -> loss: {:.7f}".format(
                epoch+1,
                num_epochs,
                iteration,
                loss.item()
            ))

    def save_model(self):
        name = self.config["model"]["features_extractor"]["save_path"]
        torch.save(self.model.save(), "{}".format(name))
        print("Saved {}".format(name))
