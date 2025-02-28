import logging

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

def collate_fn(dataset_items: list[dict]):
    waveforms = [item["data_object"] for item in dataset_items] # (time,)
    labels = torch.LongTensor([item["label"] for item in dataset_items])
    lengths = torch.LongTensor([item['length'] for item in dataset_items])
    names = [item["name"] for item in dataset_items]

    # logger.info(f'cnt_waveforms: {len(waveforms)}')
    padded_waveforms = pad_sequence(waveforms, batch_first=True)

    return {
        "data_object": padded_waveforms,
        "labels": labels,
        "lengths": lengths,
        "names" : names
    }