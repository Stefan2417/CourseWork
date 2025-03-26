import logging
import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


def move_batch_to_device(batch, device):
    """
    Move all necessary tensors to the device.

    Args:
        batch (dict): dict-based batch containing the data from
            the dataloader.
    Returns:
        batch (dict): dict-based batch containing the data from
            the dataloader with some of the tensors on the device.
    """
    for tensor_for_device in batch.keys():
        batch[tensor_for_device] = batch[tensor_for_device].to(device)
    return batch

@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)


    # build model architecture, then print to console
    model = instantiate(config.model).to(device)

    # get metrics

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path

    already_exists = save_path.exists()

    save_path.mkdir(exist_ok=True, parents=True)

    metrics = instantiate(config.metrics)

    logger.info(f'saved embeddings already exists: {already_exists}')

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=None,
        skip_model_load=config.inferencer.skip_model_load,
    )

    # if not already_exists:
    inferencer.run_inference()

    for part in inferencer.evaluation_dataloaders.keys():
        saved_directory_path = save_path / part
        logger.info(saved_directory_path)
        for metric in metrics['inference']:
            metric_value = metric(saved_directory_path)
            print(f'{metric.name}: {metric_value:.4f}\n')
            metric.reset()
    #
    # for part in logs.keys():
    #     for key, value in logs[part].items():
    #         full_key = part + "_" + key
    #         print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
