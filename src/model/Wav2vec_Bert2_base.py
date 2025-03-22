import logging
import torch
from torch import nn
from transformers import Wav2Vec2BertForXVector

logger = logging.getLogger(__name__)


class Wav2vecBert2(nn.Module):
    """

    """

    def __init__(self, pretrain):
        super().__init__()

        self.w2v = Wav2Vec2BertForXVector.from_pretrained(pretrain)

    def forward(self, batch):
        """

        """

        padded_length = batch['data_object'].shape[1]
        attention_mask = (torch.arange(padded_length)[None, :] < batch['lengths'][:, None]).long()

        embeddings = self.w2v.encode(
            input_features=batch['data_object'],
            attention_mask=attention_mask,
            output_hidden_states=True
        ).embeddings


        return {"embeddings": embeddings}