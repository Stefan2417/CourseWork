import logging

from torch import nn
import torch
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from espnet2.tasks.ssl import SSLTask


logger = logging.getLogger(__name__)

class XeusFineTunning(nn.Module):
    """
        ASV Adapter for fine-tuning from

        Liu Y., He L., Liu J., Johnson M.T. (2019)
        "Introducing Phonetic Information to Speaker Embedding for Speaker Verification"
        EURASIP Journal on Audio, Speech, and Music Processing, 2019:19.
        DOI: 10.1186/s13636-019-0166-8
        https://doi.org/10.1186/s13636-019-0166-8
    """

    def get_lr_params(self):
        # return [
        #     {'params': list(filter(lambda p: p.requires_grad, self.w2v.parameters())), 'lr': 1e-6},
        #     {'params': list(filter(lambda p: p.requires_grad,  self.layer_norm.parameters())), 'lr': 1e-4},
        #     {'params': list(filter(lambda p: p.requires_grad,  self.asp.parameters())), 'lr': 1e-4},
        #     {'params': list(filter(lambda p: p.requires_grad,  self.head.parameters())), 'lr': 1e-4},
        #     {'params': list(filter(lambda p: p.requires_grad, self.criterion.parameters())), 'lr': 1e-4}
        # ]

        params = []
        xeus_params = [p for p in self.xeus.parameters() if p.requires_grad]
        if xeus_params:
            params.append({'params': xeus_params, 'lr': 1e-6})

        layer_norm_params = [p for p in self.layer_norm.parameters() if p.requires_grad]
        if layer_norm_params:
            params.append({'params': layer_norm_params, 'lr': 1e-3})

        asp_params = [p for p in self.asp.parameters() if p.requires_grad]
        if asp_params:
            params.append({'params': asp_params, 'lr': 1e-3})

        head_params = [p for p in self.head.parameters() if p.requires_grad]
        if head_params:
            params.append({'params': head_params, 'lr': 1e-3})

        criterion_params = [p for p in self.criterion.parameters() if p.requires_grad]
        if criterion_params:
            params.append({'params': criterion_params, 'lr': 1e-3})

        return params

    def __init__(self, criterion, pretrain, emb_dim, freeze_strategy="none",
                 use_masks=False, layers=None):
        """
        """
        super().__init__()

        self.criterion = criterion

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
        elif freeze_strategy == "all":
            for param in self.xeus.parameters():
                param.requires_grad_(False)

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

        pooled = self.asp(asp_input)

        embeddings = self.head(pooled.squeeze(-1))

        return {"embeddings": embeddings}

if __name__ == "__main__":
    model = XeusFineTunning(pretrain='/home/stefan/Documents/CourseWork/XEUS/model/xeus_checkpoint.pth', emb_dim=512, layers = [5,6,7,8,9,10])
