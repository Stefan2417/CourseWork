import torch
from torch import nn
import torch.nn.functional as F


class AAMSoftmaxLoss(nn.Module):
    """
    Оптимизированная реализация AAM-Softmax (ArcFace) с улучшенной численной стабильностью и метриками.

    Параметры:
        embed_dim (int): Размер эмбеддингов
        num_classes (int): Количество классов
        scale (float): Масштаб logits (по умолчанию 30.0)
        margin (float): Угловая маржа в радианах (по умолчанию 0.4)
    """

    def __init__(self, embed_dim: int, num_classes: int, scale: float = 32.0, margin: float = 0.2):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.Tensor(num_classes, embed_dim), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

        self.cos_m = torch.cos(margin)
        self.sin_m = torch.sin(margin)
        self.threshold = torch.cos(torch.pi - margin)
        self.mm = torch.sin(torch.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
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

        with torch.no_grad():
            acc = (logits.argmax(1) == labels).float().mean()
            pos_cos = cos_theta[one_hot.bool()].mean()
            neg_cos = cos_theta[~one_hot.bool()].mean()

        return {
            "loss": loss,
            "accuracy": acc,
            "pos_cosine": pos_cos,
            "neg_cosine": neg_cos
        }
