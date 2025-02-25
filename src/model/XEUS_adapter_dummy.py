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

    def __init__(self, pretrain, layers=None, emb_dim=256):
        super().__init__()


        self.xeus, _ = SSLTask.build_model_from_file(None, model_file=pretrain)
        self.xeus.eval()
        for param in self.xeus.parameters():
            param.requires_grad_(False)


        self.selected_layers = layers
        self.num_blocks = self.num_blocks
        
        self.layer_norm = nn.LayerNorm(
            self.xeus.encoder._output_size * self.num_blocks
        )

        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(
            channels=self.xeus.encoder._output_size * self.num_blocks
        )

        # Проекционная голова
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.xeus.encoder._output_size * self.num_blocks * 2),
            nn.Linear(self.xeus.encoder._output_size * self.num_blocks * 2, emb_dim)
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
