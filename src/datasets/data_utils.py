from itertools import repeat

from hydra.utils import instantiate
from torch.utils.data import Sampler

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed
import os
import tqdm
import logging
import json

logger = logging.getLogger(__name__)


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


class DynamicBatchSampler(Sampler):
    def __init__(self, lengths, max_duration=60, sample_rate=16000):
        self.max_samples = max_duration * sample_rate
        self.lengths = lengths
        self._batches = []  # Хранит предвычисленные батчи

        # Предварительное вычисление батчей при инициализации
        indices = sorted(range(len(self.lengths)),
                         key=lambda i: self.lengths[i],
                         reverse=True)
        batch = []
        current_max = 0
        for idx in indices:
            new_max = max(current_max, self.lengths[idx])
            if new_max * (len(batch) + 1) > self.max_samples:
                self._batches.append(batch)
                batch = [idx]
                current_max = self.lengths[idx]
            else:
                batch.append(idx)
                current_max = new_max
        if batch:
            self._batches.append(batch)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def get_dataloaders(config, device):
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    datasets = instantiate(config.datasets)
    dataloaders = {}

    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        dataset_lengths = [item["length"] for item in dataset]

        dynamic_sampler = DynamicBatchSampler(
            lengths=dataset_lengths,
            max_duration=dataset.max_samples // dataset.sample_rate,
            sample_rate=dataset.sample_rate
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            batch_sampler=dynamic_sampler,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed
        )

        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms


def parse_dataset_speakers(dir_name, extension='.wav', speaker_level=1):
    if dir_name[-1] != os.sep:
        dir_name += os.sep

    prefix_size = len(dir_name)
    speakers_data = {}

    logger.info("finding {}, Waiting...".format(extension))

    for root, dirs, filenames in tqdm.tqdm(os.walk(dir_name, followlinks=True)):
        filtered_files = [os.path.join(root, f) for f in filenames if f.endswith(extension)]
        if len(filtered_files) > 0:
            speaker_name = (os.sep).join(root[prefix_size:].split(os.sep)[:speaker_level])
            if speaker_name not in speakers_data:
                speakers_data[speaker_name] = []
            speakers_data[speaker_name] += filtered_files

    return speakers_data


def extract_speaker_id(speaker_id: str) -> int:
    if not speaker_id.startswith('id'):
        raise ValueError("Invalid speaker ID format")

    number_part = speaker_id[2:]
    if not number_part.isdigit():
        raise ValueError("ID contains non-numeric characters")

    return int(number_part)