import logging
import torch
from torch import nn
from transformers import Wav2Vec2BertForXVector

logger = logging.getLogger(__name__)


class Wav2vecBert2Base(nn.Module):
    """

    """

    def __init__(self, pretrain="facebook/w2v-bert-2.0"):
        super().__init__()
        self.w2v = Wav2Vec2BertForXVector.from_pretrained(pretrain)
        for param in self.w2v.parameters():
            param.requires_grad = False

        for param in self.w2v.projector.parameters():
            param.requires_grad = True

        for param in self.w2v.tdnn.parameters():
            param.requires_grad = True
        for param in self.w2v.feature_extractor.parameters():
            param.requires_grad = True
        for param in self.w2v.classifier.parameters():
            param.requires_grad = True

    def forward(self, batch):
        """

        """

        embeddings = self.w2v(
            input_features=batch['data_object'],
            attention_mask=batch['lengths'],
        ).embeddings

        return {"embeddings": embeddings}