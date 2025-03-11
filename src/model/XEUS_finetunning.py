from torch import nn
import torch
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d
from espnet2.tasks.ssl import SSLTask


class XeusFineTunning(nn.Module):
    """
    """

    def __init__(self, pretrain, emb_dim, freeze_strategy="none",
                 trainable_layers=None, layers_to_use=None):
        """
        """
        super().__init__()

        self.xeus, _ = SSLTask.build_model_from_file(
            None,
            model_file=pretrain
        )

        num_layers = len(self.xeus.encoders.encoders)

        if layers_to_use is None:
            layers_to_use = [0, num_layers // 2, num_layers - 1]

        self.selected_layers = layers_to_use
        self.num_blocks = len(self.selected_layers)
        self.output_xeus_emb_sz = 1024

        if freeze_strategy == "none":
            for param in self.xeus.parameters():
                param.requires_grad_(True)
        elif freeze_strategy == "partial":
            for param in self.xeus.parameters():
                param.requires_grad_(False)

            if trainable_layers:
                for idx in trainable_layers:
                    if 0 <= idx < num_layers:
                        for param in self.xeus.encoders.encoders[idx].parameters():
                            param.requires_grad_(True)

        self.layer_norm = nn.LayerNorm(
            self.output_xeus_emb_sz * self.num_blocks
        )

        self.asp = AttentiveStatisticsPooling(
            channels=self.output_xeus_emb_sz * self.num_blocks
        )

        self.head = nn.Sequential(
            BatchNorm1d(input_size=self.output_xeus_emb_sz * self.num_blocks * 2),
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
