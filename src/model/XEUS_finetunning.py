import logging

from torch import nn
import torch
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d
from espnet2.tasks.ssl import SSLTask


logger = logging.getLogger(__name__)

class XeusFineTunning(nn.Module):
    """
    """

    def __init__(self, pretrain, emb_dim, freeze_strategy="none",
                 use_masks=False, layers=None):
        """
        """
        super().__init__()

        self.xeus, _ = SSLTask.build_model_from_file(
            None,
            model_file=pretrain
        )

        for layer in self.xeus.encoder.encoders:
            layer.use_flash_attn = True

        if use_masks:
            self.xeus.masker.mask_prob = 0.65  # default 0.8
            self.xeus.masker.mask_length = 20  # default 10
            self.xeus.masker.mask_selection = 'static'  # default 'uniform'

        assert layers is not None

        self.selected_layers = layers
        self.num_blocks = len(self.selected_layers)
        self.output_xeus_emb_sz = 1024
        if freeze_strategy == 'only_encoders':
            for param in self.xeus.parameters():
                param.requires_grad_(False)
            for encoder in self.xeus.encoder.encoders:
                for param in encoder.parameters():
                    param.requires_grad_(True)
        elif freeze_strategy == "none":
            pass
        elif freeze_strategy == "partial":
            for param in self.xeus.parameters():
                param.requires_grad_(False)

            num_xeus_layers = len(self.xeus.encoder.encoders)

            layers_to_unfreeze = num_xeus_layers // 2
            for i in range(num_xeus_layers - layers_to_unfreeze, num_xeus_layers):
                print(f'unfreeze: {i}')
                for param in self.xeus.encoder.encoders[i].parameters():
                    param.requires_grad_(True)

        self.layer_norm = nn.LayerNorm(
            self.output_xeus_emb_sz * self.num_blocks
        )

        self.asp = AttentiveStatisticsPooling(
            channels=self.output_xeus_emb_sz * self.num_blocks
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.output_xeus_emb_sz * self.num_blocks * 2),
            nn.Linear(self.output_xeus_emb_sz * self.num_blocks * 2, emb_dim)
        )

    def forward(self, batch):
        encoder_outputs = self.xeus.encode(
            batch['data_object'],
            batch['lengths'],
            use_mask=False,
            use_final_output=False
        )[0]

        selected_features = [encoder_outputs[layer_idx] for layer_idx in self.selected_layers]

        concatenated = torch.cat(selected_features, dim=-1)  # [batch, time, dim*N]

        normalized = self.layer_norm(concatenated)

        asp_input = normalized.permute(0, 2, 1)

        pooled = self.asp(asp_input, lengths=batch['lengths'])

        embeddings = self.head(pooled.squeeze(-1))

        return {"embeddings": embeddings}
