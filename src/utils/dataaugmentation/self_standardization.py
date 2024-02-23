import torch


class SelfStandardization(torch.nn.Module):
    def forward(self, x):
        mean = x.mean(dim=[1, 2]).unsqueeze(1).unsqueeze(2)
        std = x.std(dim=[1, 2]).unsqueeze(1).unsqueeze(2)
        eps = torch.finfo(torch.float32).eps
        x = (x - mean) / (std + eps)
        return x
