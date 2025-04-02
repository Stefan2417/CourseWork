import logging
import torch
from torch import nn
from transformers import Wav2Vec2BertModel, Wav2Vec2BertProcessor
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d


class Wav2vecBert2Adapter(nn.Module):

    def get_lr_params(self):

        params = []
        w2v_params = [p for p in self.w2v.parameters() if p.requires_grad]
        if w2v_params:
            params.append({'params': w2v_params, 'lr': self.main_lr})

        layer_norm_params = [p for p in self.layer_norm.parameters() if p.requires_grad]
        if layer_norm_params:
            params.append({'params': layer_norm_params, 'lr': self.adapter_lr})

        asp_params = [p for p in self.asp.parameters() if p.requires_grad]
        if asp_params:
            params.append({'params': asp_params, 'lr': self.adapter_lr})

        head_params = [p for p in self.head.parameters() if p.requires_grad]
        if head_params:
            params.append({'params': head_params, 'lr': self.adapter_lr})

        criterion_params = [p for p in self.criterion.parameters() if p.requires_grad]
        if criterion_params:
            params.append({'params': criterion_params, 'lr': self.adapter_lr})

        return params

    def __init__(self, criterion, emb_dim, pretrained, freeze_strategy="none", layers=None, main_lr=0, adapter_lr=0):
        super().__init__()

        self.w2v = Wav2Vec2BertModel.from_pretrained(
            'facebook/w2v-bert-2.0', cache_dir=pretrained
        )

        self.criterion = criterion
        self.main_lr = main_lr
        self.adapter_lr = adapter_lr

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

        encoder_outputs = self.w2v(
            input_features=batch['data_object'],
            attention_mask=batch['lengths'],
            output_hidden_states=True
        )

        hidden_states = encoder_outputs.hidden_states[1:]

        selected_features = [hidden_states[layer_idx + 1] for layer_idx in self.selected_layers]

        concatenated = torch.cat(selected_features, dim=-1)

        normalized = self.layer_norm(concatenated)

        asp_input = normalized.permute(0, 2, 1)

        pooled = self.asp(asp_input)

        embeddings = self.head(pooled.squeeze(-1))

        return {"embeddings": embeddings}
