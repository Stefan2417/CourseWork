from itertools import repeat

from hydra.utils import instantiate

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



def get_dataloaders(config, device):
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    datasets = instantiate(config.datasets)
    dataloaders = {}
    logger.info('get dataloaders')

    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            worker_init_fn=set_worker_seed,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
        )

        dataloaders[dataset_partition] = partition_dataloader
        logger.info(f'success instantiate dataset partition {dataset_partition}')

    return dataloaders, batch_transforms


def parse_dataset_speakers(dir_name, extension, speaker_level=1):
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