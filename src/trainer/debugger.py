import torch
from tqdm.auto import tqdm

from src.trainer.base_trainer import BaseTrainer


class Debugger(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
            self,
            model,
            config,
            device,
            dataloaders,
            writer,
            logger,
            batch_transforms=None,
            skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
                skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.debugger

        self.device = device

        self.logger = logger

        self.model = model
        self.batch_transforms = batch_transforms

        self.epochs = self.config.debugger.epochs
        self.cur_step = 0

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        self.writer = writer

        if not skip_model_load: #TODO
        # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_debug(self):
        """
        Run inference on each partition.

        Returns: nothing
        """

        for part, dataloader in self.evaluation_dataloaders.items():
            self._debug_part(part, dataloader)

    def process_batch(self, batch):
        """
        Run batch through the model and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns: nothing, batch must be saved
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(batch)
        batch.update(outputs)

        return batch

    def _debug_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()
        for epoch in range (1, self.epochs + 1):
            with torch.no_grad():
                for batch_idx, batch in tqdm(
                        enumerate(dataloader),
                        desc=part,
                        total=len(dataloader),
                ):
                    self.process_batch(
                        batch=batch
                    )

                    self._log_batch(batch_idx, batch)

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        for ind in range(len(batch['data_object'])):
            self.writer.set_step(self.cur_step)
            self.cur_step += 1
            self.writer.add_audio(audio_name=batch['names'][ind], audio=batch['data_object'][ind],
                                  sample_rate=self.config.writer.sample_rate)
