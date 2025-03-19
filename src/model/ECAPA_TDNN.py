import logging

import torch
from torch import nn
from speechbrain.inference.speaker import EncoderClassifier

logger = logging.getLogger(__name__)

class EmbedderRawNet(nn.Module):
    """
    """

    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = EncoderClassifier.from_hparams(
            source="yangwang825/ecapa-tdnn-vox2",
            savedir="pretrained/ecapa_vox2",
            run_opts={"device": self.device}
        )

    def forward(self, batch: dict) -> torch.Tensor:
        waveforms = batch['data_object']

        lengths = batch['lengths'].float()
        lengths /= lengths.max()

        embeddings = self.model.encode_batch(wavs=waveforms, wav_lens=lengths)

        return {'embeddings' : embeddings.squeeze(1)}

