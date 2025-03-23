import logging
import warnings

import hydra
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Debugger
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
import os
from src.utils.init_utils import set_random_seed, setup_saving_and_logging_debug


# warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="debug")
def main(config):
    os.environ["WANDB_API_KEY"] = "8dd6bf64e54bcd000df67566620fe0a54f6ce31a"  # TODO - delete token
    set_random_seed(config.debugger.seed)

    if config.debugger.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.debugger.device

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging_debug(config)
    writer = instantiate(config.writer, logger, project_config)

    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)

    debugger = Debugger(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        writer=writer,
        skip_model_load=config.debugger.skip_model_load,
        logger=logger
    )

    debugger.run_debug()


if __name__ == "__main__":
    main()
