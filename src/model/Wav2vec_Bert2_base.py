import logging
import torch
from torch import nn
from transformers import Wav2Vec2BertForXVector

logger = logging.getLogger(__name__)


class Wav2vecBert2Base(nn.Module):
    """

    """

    def get_lr_params(self):
        params = []

        wav2vec2_bert_params = [p for p in self.w2v.wav2vec2_bert.parameters() if p.requires_grad]
        if wav2vec2_bert_params:
            params.append({'params': wav2vec2_bert_params, 'lr': 1e-6})

        projector_params = [p for p in self.w2v.projector.parameters() if p.requires_grad]
        if projector_params:
            params.append({'params': projector_params, 'lr': 1e-4})

        tdnn_params = [p for p in self.w2v.tdnn.parameters() if p.requires_grad]
        if tdnn_params:
            params.append({'params': tdnn_params, 'lr': 1e-4})

        feature_extractor_params = [p for p in self.w2v.feature_extractor.parameters() if p.requires_grad]
        if feature_extractor_params:
            params.append({'params': feature_extractor_params, 'lr': 1e-4})

        classifier_params = [p for p in self.w2v.classifier.parameters() if p.requires_grad]
        if classifier_params:
            params.append({'params': classifier_params, 'lr': 1e-4})

        criterion_params = [p for p in self.criterion.parameters() if p.requires_grad]
        if criterion_params:
            params.append({'params': criterion_params, 'lr': 1e-4})

        return params

    def __init__(self, criterion, pretrained):
        super().__init__()
        self.criterion = criterion

        self.w2v = Wav2Vec2BertForXVector.from_pretrained('facebook/w2v-bert-2.0', cache_dir=pretrained)

    def forward(self, batch):
        """

        """

        embeddings = self.w2v(
            input_features=batch['data_object'],
            attention_mask=batch['lengths'],
        ).embeddings

        return {"embeddings": embeddings}