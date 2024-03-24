import os
from datetime import datetime

import torch


class SaveCheckpoints:
    def __init__(self, path):
        self.path = path

        self.name = None
        self.directory_name = None

        self.initialize()

    def initialize(self):
        self.name = self.path.split("/")[-1].strip()
        self.create_directory()

    def create_directory(self):
        self.directory_name = "{}_{}".format(self.path, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.mkdir(path=self.directory_name)

    def save(self, model, epoch):
        save_path = "{}/{}_{}.{}".format(self.directory_name, self.name, epoch, "pt")
        torch.save(model.save(), save_path)
        print("Saved checkpoints {}".format(save_path))

