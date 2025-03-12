import torch
from torch import nn
import torch.nn.functional as F
import math


class AAMSoftmaxLoss(nn.Module):
    """
    args:
        embed_dim (int):
        num_classes (int):
        scale (float):
        margin (float):
    """

    def __init__(self, embed_dim: int, num_classes: int, scale: float = 32.0, margin: float = 0.2):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.Tensor(num_classes, embed_dim), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, batch) -> dict:

        embeddings = batch['embeddings']
        labels = batch['labels']

        embeddings = embeddings.float()

        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weights_norm = F.normalize(self.weight, p=2, dim=1)

        cos_theta = F.linear(embeddings_norm, weights_norm)

        sine = torch.sqrt(torch.clamp(1.0 - cos_theta.pow(2), min=1e-6))
        phi = cos_theta * self.cos_m - sine * self.sin_m

        phi = torch.where(cos_theta > self.threshold, phi, cos_theta - self.mm)

        one_hot = F.one_hot(labels, num_classes=self.num_classes)

        logits = torch.where(one_hot.bool(), phi, cos_theta)
        logits *= self.scale

        loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss
        }
