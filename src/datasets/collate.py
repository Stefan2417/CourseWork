import logging

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

def collate_fn(dataset_items: list[dict]):
    waveforms, labels, lengths, names = [], [], [], []

    for item in dataset_items:
        waveforms.append(item["data_object"])
        labels.append(item["label"])
        lengths.append(item['length'])
        names.append(item["name"])

    padded_waveforms = pad_sequence(waveforms, batch_first=True)

    return {
        "data_object": padded_waveforms,
        "labels": labels,
        "lengths": lengths,
        "names" : names
    }