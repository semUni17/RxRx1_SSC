import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2


class SimCLR(nn.Module):
    def __init__(self, features_extractor, projection_head, temperature=0.5, dist=None):
        super().__init__()
        self.features_extractor = features_extractor
        self.projection_head = projection_head
        self.temperature = temperature
        self.dist = dist

        self.initialize()

    def initialize(self):
        pass

    def encode(self, x):
        h = self.features_extractor(x)
        return h

    def project(self, h):
        z = self.projection_head(h)
        z = F.normalize(z)
        return z

    def infonce_loss(self, n, z_i, z_j):
        if self.dist is None:
            zis = [z_i]
            zjs = [z_j]
        else:
            zis = [torch.zeros_like(z_i) for _ in range(self.dist.get_world_size())]
            zjs = [torch.zeros_like(z_j) for _ in range(self.dist.get_world_size())]
            self.dist.all_gather(zis, z_i)
            self.dist.all_gather(zjs, z_j)

        z1 = torch.cat((z_i, z_j), dim=0)
        z2 = torch.cat(zis + zjs, dim=0)

        sim_matrix = torch.mm(z1, z2.t())
        sim_matrix = sim_matrix / self.temperature
        n_gpus = 1 if self.dist is None else self.dist.get_world_size()
        rank = 0 if self.dist is None else self.dist.get_rank()
        sim_matrix[torch.arange(n), torch.arange(rank*n, (rank+1)*n)] = -float('inf')
        sim_matrix[torch.arange(n, 2*n), torch.arange((n_gpus+rank)*n, (n_gpus+rank+1)*n)] = -float('inf')

        targets = torch.cat((torch.arange((n_gpus+rank)*n, (n_gpus+rank+1)*n), torch.arange(rank*n, (rank+1)*n)), dim=0)
        targets = targets.to(sim_matrix.get_device()).long()

        loss = F.cross_entropy(sim_matrix, targets, reduction="sum")
        loss = loss / n

        return loss

    def forward(self, x):
        n = x[0].shape[0]

        x_i, x_j = x
        h_i, h_j = self.encode(x_i), self.encode(x_j)
        z_i, z_j = self.project(h_i), self.project(h_j)

        loss = self.infonce_loss(n, z_i, z_j)

        '''xi_0 = x_i[0].detach().cpu().numpy().transpose(1, 2, 0)
        xi_1 = x_i[1].detach().cpu().numpy().transpose(1, 2, 0)
        xi_aug = np.concatenate((xi_0, xi_1), axis=1)
        # cv2.imshow("xi_aug", xi_aug)
        xj_0 = x_j[0].detach().cpu().numpy().transpose(1, 2, 0)
        xj_1 = x_j[1].detach().cpu().numpy().transpose(1, 2, 0)
        xj_aug = np.concatenate((xj_0, xj_1), axis=1)
        # cv2.imshow("xj_aug", xj_aug)
        total = np.concatenate((xi_aug, xj_aug), axis=0)
        cv2.imshow("total", total)
        cv2.waitKey(500)'''

        return loss

    def save(self):
        return self.features_extractor.save()
