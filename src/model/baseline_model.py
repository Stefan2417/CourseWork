from torch import nn
from torch.nn import Sequential
import torch
from espnet2.tasks.ssl import SSLTask


class XeusEmbedder(nn.Module):
    """
    Xeus-based audio embedding extractor with temporal pooling
    """

    def __init__(self, pretrain):
        """
        Args:
            xeus_model (nn.Module): pretrained Xeus model
            pool_method (str): temporal pooling method (mean/max)
        """
        super().__init__()
        self.xeus, _ = SSLTask.build_model_from_file(
            None,
            model_file=pretrain
        )
        self.xeus.eval()
        for param in self.xeus.parameters():
            param.requires_grad_(False)

    def forward(self, batch):
        """
        Processing pipeline:
        1. Feature extraction with Xeus
        2. Chunk splitting by durations
        3. Temporal pooling

        Args:
            waveforms (Tensor): raw audio [batch, time]
            wav_lens (Tensor): chunk lengths for each sample

        Returns:
            embeddings (Tensor): pooled embeddings [batch, feat_dim]
        """
        data_object = batch['data_object']
        lengths = batch['lengths']
        # Feature extraction
        feats = self.xeus.encode(
            data_object,
            lengths,
            use_mask=False,
            use_final_output=False
        )[0][-1]

        # pooled = torch.mean(feats, dim=1)
        mu = feats.mean(dim=1)
        std = feats.std(dim=1)
        pooled = torch.cat([mu, std], dim=1)

        return {"embeddings": pooled}

    def __str__(self):
        """Информация о модели"""
        stats = [
            f"Xeus model: {self.xeus.__class__.__name__}",
            f"Frozen params: {sum(p.numel() for p in self.xeus.parameters())}",
            f"Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        ]
        return "\n".join(stats)

