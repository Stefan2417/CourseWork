import logging
import torch
from torch import nn
from transformers import Wav2Vec2BertModel, Wav2Vec2BertProcessor
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d

logger = logging.getLogger(__name__)


class Wav2vecBert2Adapter(nn.Module):
    """

    """

    def __init__(self, emb_dim, pretrain="facebook/w2v-bert-2.0", freeze_strategy="none", layers=None):
        super().__init__()

        self.w2v = Wav2Vec2BertModel.from_pretrained(
            pretrain,
        )

        assert layers is not None

        self.selected_layers = layers
        self.num_blocks = len(self.selected_layers)
        self.output_w2v_emb_sz = self.w2v.config.hidden_size

        self.layer_norm = nn.LayerNorm(
            self.output_w2v_emb_sz * self.num_blocks
        )

        self.asp = AttentiveStatisticsPooling(
            channels=self.output_w2v_emb_sz * self.num_blocks
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.output_w2v_emb_sz * self.num_blocks * 2),
            nn.Linear(self.output_w2v_emb_sz * self.num_blocks * 2, emb_dim)
        )
        if freeze_strategy == "none":
            pass
        if freeze_strategy == "all":
            for param in self.w2v.parameters():
                param.requires_grad = False

    def forward(self, batch):
        """

        """

        encoder_outputs = self.w2v(
            input_features=batch['data_object'],
            attention_mask=batch['lengths'],
            output_hidden_states=True
        )

        hidden_states = encoder_outputs.hidden_states

        selected_features = [hidden_states[layer_idx + 1] for layer_idx in self.selected_layers]

        concatenated = torch.cat(selected_features, dim=-1)

        normalized = self.layer_norm(concatenated)

        asp_input = normalized.permute(0, 2, 1)

        pooled = self.asp(asp_input, lengths=batch['lengths'])

        embeddings = self.head(pooled.squeeze(-1))

        return {"embeddings": embeddings}