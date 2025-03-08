from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if self.is_train:
            self.optimizer.zero_grad()

        outputs = self.model(batch)
        batch.update(outputs)

        if self.is_train:
            all_losses = self.criterion(batch)
            batch.update(all_losses)

            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            for loss_name in self.config.writer.loss_names:
                metrics.update(loss_name, batch[loss_name].item())

        if not self.is_train:
            metric_funcs = self.metrics["train_inference"]
            for met in metric_funcs:
                met(batch_embeddings=batch)
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """

        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics
                )

            metric_funcs = self.metrics["train_inference"]
            for met in metric_funcs:
                self.evaluation_metrics.update(met.name, met())  # calc EER

            self.writer.set_step(self.cur_step, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

        return self.evaluation_metrics.result()

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

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            if self.log_audio:
                self.writer.add_audio(audio_name='audio_sample_train', audio=batch['data_object'][0],
                                      sample_rate=self.config.writer.sample_rate)
        else:
            # Log Stuff
            if self.log_audio:
                self.writer.add_audio(audio_name='audio_sample_eval', audio=batch['data_object'][0],
                                      sample_rate=self.config.writer.sample_rate)
