import logging
import torch
from torch import nn
from transformers import Wav2Vec2BertForXVector

logger = logging.getLogger(__name__)


class Wav2vecBert2Base(nn.Module):
    """

    """

    def get_lr_params(self):
        return [
            {'params': [p for p in self.w2v.wav2vec2_bert.parameters() if p.requires_grad], 'lr': 1e-6},
            {'params': [p for p in self.w2v.projector.parameters() if p.requires_grad], 'lr': 1e-4},
            {'params': [p for p in self.w2v.tdnn.parameters() if p.requires_grad], 'lr': 1e-4},
            {'params': [p for p in self.w2v.feature_extractor.parameters() if p.requires_grad], 'lr': 1e-4},
            {'params': [p for p in self.w2v.classifier.parameters() if p.requires_grad], 'lr': 1e-4},
        ]

    def __init__(self, pretrained):
        super().__init__()
        self.w2v = Wav2Vec2BertForXVector.from_pretrained('facebook/w2v-bert-2.0', cache_dir=pretrained)

    def forward(self, batch):
        """

        """

        embeddings = self.w2v(
            input_features=batch['data_object'],
            attention_mask=batch['lengths'],
        ).embeddings

        return {"embeddings": embeddings}