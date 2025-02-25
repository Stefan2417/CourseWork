from torch import nn
from espnet2.bin.spk_inference import SpeakerEmbedding



class ECAPA_TDNN(nn.Module):
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
        self.embedder = SpeakerEmbedding(
            model_tag="ecapa_tdnn",
        )
        self.embedder.eval()

    def forward(self, batch):
        """

        """
        data_object = batch['data_object']
        lengths = batch['lengths']
        # Feature extraction

        feats = self.embedder(data_object, lengths)

        return {"embeddings": feats}

    def __str__(self):
        """Информация о модели"""
        return ""

