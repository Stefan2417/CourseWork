import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from transformers import AutoFeatureExtractor

logger = logging.getLogger(__name__)

def collate_fn(dataset_items: list[dict]):
    waveforms, labels, lengths, names = [], [], [], []

    for item in dataset_items:
        waveforms.append(item["data_object"])
        labels.append(item["label"])
        lengths.append(item['length'])
        names.append(item["name"])

    labels = torch.LongTensor(labels)
    lengths = torch.LongTensor(lengths)

    padded_waveforms = pad_sequence(waveforms, batch_first=True)

    return {
        "data_object": padded_waveforms,
        "labels": labels,
        "lengths": lengths,
        "names" : names
    }

class CollateW2V(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    def forward(self, dataset_items: list[dict]):
        waveforms, labels, lengths, names = [], [], [], []
        for item in dataset_items:
            waveforms.append(item["data_object"])
            labels.append(item["label"])
            names.append(item["name"])

        extracted = self.extractor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
        return {
            "data_object" : extracted['input_features'],
            "lengths" : extracted['attention_mask'],
            "labels": labels,
            "names": names,
        }
