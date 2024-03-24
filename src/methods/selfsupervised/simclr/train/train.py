import yaml

import torch
from torchvision.transforms import *
from torch.optim import *
from torch.cuda.amp import GradScaler

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

from src.methods.classifier.model.features_extractor import FeaturesExtractor
from src.methods.selfsupervised.simclr.model.projection_head import ProjectionHead
from src.methods.selfsupervised.simclr.model.simclr import SimCLR
from src.methods.selfsupervised.simclr.dataaugmentation.multi_view_generator import MultiViewGenerator
from src.utils.dataaugmentation.self_standardization import SelfStandardization

from src.utils.checkpoints.save_checkpoints import SaveCheckpoints


class Train:
    def __init__(self, config_path: str):
        self.config_path = config_path

        self.config = None
        self.device = None
        self.transform = None
        self.train_dataset = None
        self.train_dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.save_checkpoint = None

        self.initialize()

    def initialize(self):
        self.define_config()
        self.define_transform()
        self.define_dataset()
        self.define_dataloader()
        self.define_model()
        self.define_optimizer()

        self.save_checkpoint = SaveCheckpoints(self.config["model"]["features_extractor"]["save_path"])

    def define_config(self):
        with open(self.config_path, "r") as stream:
            self.config = yaml.safe_load(stream)

        for k, v in self.config.items():
            print("{}: {}".format(k, v))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def define_transform(self):
        size = self.config["data_augmentation"]["general"]["size"]
        scaler = self.config["data_augmentation"]["color_jitter"]["scaler"]

        color_jitter = ColorJitter(0.8*scaler, 0.8*scaler, 0.8*scaler, 0.2*scaler)
        gaussian_blur = GaussianBlur(kernel_size=self.config["data_augmentation"]["gaussian_blur"]["kernel_size"])

        self.transform = Compose([
            ToTensor(),
            Resize(size=size),
            RandomResizedCrop(size=size),
            RandomErasing(scale=self.config["data_augmentation"]["erasing"]["scale"]),
            RandomRotation(degrees=self.config["data_augmentation"]["rotation"]["degrees"]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomApply([color_jitter], p=self.config["data_augmentation"]["color_jitter"]["probability"]),
            RandomApply([gaussian_blur], p=self.config["data_augmentation"]["gaussian_blur"]["probability"]),
            RandomGrayscale(p=self.config["data_augmentation"]["grayscale"]["probability"]),
            SelfStandardization()
        ])

    def define_dataset(self):
        dataset = get_dataset(
            dataset=self.config["dataset"]["name"],
            download=self.config["dataset"]["download"]
        )
        self.train_dataset = dataset.get_subset(
            split=self.config["dataset"]["split"],
            frac=self.config["dataset"]["fraction"],
            transform=MultiViewGenerator(self.transform),
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
        self.model = SimCLR(features_extractor, projection_head, self.config["hyper_parameters"]["temperature"])
        self.model.to(self.device)
        self.model.train()

    def define_optimizer(self):
        self.optimizer = SGD(
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

                loss = self.model(images)
                self.scaler.scale(loss).backward()

                if ((iteration+1) % self.config["hyper_parameters"]["gradient_accumulation"]) == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    self.print_status(epoch, num_epochs, iteration, loss)

            if ((iteration+1) % self.config["hyper_parameters"]["gradient_accumulation"]) != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            self.checkpoints(epoch)
        self.checkpoints("final")

    @staticmethod
    def print_status(epoch, num_epochs, iteration, loss):
        print("Epoch {:03d}/{:03d} ({:06d}-th iteration) -> loss: {:.7f}".format(
            epoch+1,
            num_epochs,
            iteration,
            loss.item()
        ))

    def checkpoints(self, epoch):
        if not isinstance(epoch, str):
            if (epoch % 10) == 0:
                self.save_checkpoint.save(self.model, epoch)
        elif epoch == "final":
            self.save_checkpoint.save(self.model, epoch)
