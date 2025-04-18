import warnings

import hydra
import torch
import os

from comet_ml.file_uploader import total_len
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="bebra")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    os.environ["WANDB_API_KEY"] = config.wandb_api_key

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    logger.info('run get_dataloaders')
    dataloaders, batch_transforms = get_dataloaders(config, device)

    logger.info('instantiate dataloaders and batch_transforms')
    # build model architecture, then print to console

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    model = instantiate(config.model, criterion=loss_function).to(device)

    logger.info('instantiate model')

    total_length = len(dataloaders['train']) * config.trainer.n_epochs
    logger.info(dataloaders.keys())
    logger.info(f'total_length: {total_length}')

    # build optimizer, learning rate scheduler
    optimizer = torch.optim.Adam(params = model.get_lr_params(), weight_decay=config.optimizer.weight_decay)

    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    scaler = torch.GradScaler()

    epoch_len = config.trainer.get("epoch_len", None)

    trainer = Trainer(
        model=model,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        scaler=scaler
    )

    trainer.train()


if __name__ == "__main__":
    main()
