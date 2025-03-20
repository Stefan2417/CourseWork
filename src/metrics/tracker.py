import pandas as pd
import torch
import accelerate

class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter): WandB experiment tracker with accelerator
        """
        self.writer = writer
        if writer is not None:
            self.accelerator = writer.accelerator
        else:
            raise ValueError("WandBWriter is required for this implementation of MetricTracker")

        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        if not isinstance(value, torch.Tensor):
            value_tensor = torch.tensor(value)
        else:
            value_tensor = value

        gathered_value = self.accelerator.gather_for_metrics(value_tensor)

        if self.accelerator.is_main_process:
            mean_value = gathered_value.mean().item()
            self._data.loc[key, "total"] += mean_value * n
            self._data.loc[key, "counts"] += n
            self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]
            self.writer.add_scalar(key, mean_value)

    def avg(self, key):
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        return self._data.average[key]

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        return dict(self._data.average)

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.total.keys()
