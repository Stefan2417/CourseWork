from torch import nn
import torch
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from espnet2.tasks.ssl import SSLTask


class XeusASVAdapter(nn.Module):
    """
    ASV Adapter for fine-tuning from

    Liu Y., He L., Liu J., Johnson M.T. (2019)
    "Introducing Phonetic Information to Speaker Embedding for Speaker Verification"
    EURASIP Journal on Audio, Speech, and Music Processing, 2019:19.
    DOI: 10.1186/s13636-019-0166-8
    https://doi.org/10.1186/s13636-019-0166-8
    """

    def __init__(self, pretrain, emb_dim, layers=None):
        super().__init__()

        import torch

        # Loading xeus model with flash attention configuration
        self.xeus, _ = SSLTask.build_model_from_file(
            None,
            model_file=pretrain
        )

        for layer in self.xeus.encoder.encoders:
            layer.use_flash_attn = True

        for param in self.xeus.parameters():
            param.requires_grad_(False)

        self.selected_layers = layers
        self.num_blocks = len(self.selected_layers)
        self.output_xeus_emb_sz = 1024

        self.layer_norm = nn.LayerNorm(
            self.output_xeus_emb_sz * self.num_blocks
        )

        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(
            channels=self.output_xeus_emb_sz * self.num_blocks
        )

        # Проекционная голова
        self.head = nn.Sequential(
            nn.LayerNorm(self.output_xeus_emb_sz * self.num_blocks * 2),
            nn.Linear(self.output_xeus_emb_sz * self.num_blocks * 2, emb_dim)
        )

    def forward(self, batch):
        # Enable flash attention for the forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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
