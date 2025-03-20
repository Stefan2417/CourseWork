from datetime import datetime

import numpy as np
import pandas as pd
from accelerate import Accelerator
import wandb
import os


class WandBWriter:
    """
    Class for experiment tracking via WandB.

    See https://docs.wandb.ai/.
    """

    def __init__(
            self,
            accelerator : Accelerator,
            logger,
            project_config,
            project_name,
            entity=None,
            run_id=None,
            run_name=None,
            mode="online",
            **kwargs,
    ):
        """
        API key is expected to be provided by the user in the terminal.

        Args:
            logger (Logger): logger that logs output.
            project_config (dict): config for the current experiment.
            project_name (str): name of the project inside experiment tracker.
            entity (str | None): name of the entity inside experiment
                tracker. Used if you work in a team.
            run_id (str | None): the id of the current run.
            run_name (str | None): the name of the run. If None, random name
                is given.
            mode (str): if online, log data to the remote server. If
                offline, log locally.
        """
        self.accelerator = accelerator
        api_key = os.getenv("WANDB_API_KEY")

        if api_key:
            wandb.login(key=api_key)
        else:
            logger.error("WANDB_API_KEY not found")

        self.run_id = run_id

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=project_name,
                config=project_config,
                init_kwargs={
                    "wandb": {
                        "entity": entity,
                        "name": run_name,
                        "resume": "allow",
                        "id": self.run_id,
                        "mode": mode,
                        "save_code": kwargs.get("save_code", False),
                    }
                }
            )

        self.step = 0
        # the mode is usually equal to the current partition name
        # used to separate Partition1 and Partition2 metrics
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor
        training/evaluation speed.

        Args:
            step (int): current step.
            mode (str): current mode (partition name).
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def get_step(self):
        return self.step

    def _object_name(self, object_name):
        """
        Update object_name (scalar, image, etc.) with the
        current mode (partition name). Used to separate metrics
        from different partitions.

        Args:
            object_name (str): current object name.
        Returns:
            object_name (str): updated object name.
        """
        return f"{object_name}_{self.mode}"

    def _log(self, data):
        self.accelerator.log(data, step=self.step)

    def add_scalar(self, scalar_name, scalar):
        self._log({self._object_name(scalar_name): scalar})

    def add_image(self, image_name, image):
        self._log({self._object_name(image_name): wandb.Image(image)})

    def add_audio(self, audio_name, audio, sample_rate=None):
        audio_np = audio.detach().cpu().numpy().T
        self._log({
            self._object_name(audio_name): wandb.Audio(audio_np, sample_rate=sample_rate)
        })
