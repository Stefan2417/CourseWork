import logging

from torch.utils.data import Sampler
import random


logger = logging.getLogger(__name__)

class DynamicBatchSampler(Sampler):
    def __init__(self, shuffle : bool, max_duration : int, lengths : list, sample_rate : int):
        super().__init__()
        self.max_samples = max_duration * sample_rate
        self.lengths = lengths
        self.shuffle = shuffle
        random.seed(42)
        self.len = -1

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        batch = []
        current_max = 0
        for idx in indices:
            new_max = max(current_max, self.lengths[idx])
            if new_max * (len(batch) + 1) > self.max_samples:

                yield batch
                batch = [idx]
                current_max = self.lengths[idx]
            else:
                batch.append(idx)
                current_max = new_max
        if batch:
            yield batch

    def __len__(self):
        if self.len != -1:
            return self.len
        self.len = sum(1 for _ in self.__iter__())
        return self.len