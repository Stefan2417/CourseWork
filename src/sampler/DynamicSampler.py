import logging

from torch.utils.data import Sampler
import random
import torch
import torch.nn as nn
import random
import math
from speechbrain.inference.VAD import VAD


logger = logging.getLogger(__name__)

# class DynamicBatchSampler(Sampler):
#     def __init__(self, shuffle : bool, max_duration : int, lengths : list, sample_rate : int):
#         super().__init__()
#         self.max_samples = max_duration * sample_rate
#         self.lengths = lengths
#         self.shuffle = shuffle
#         random.seed(42)
#         self.len = -1
#
#     def __iter__(self):
#         indices = list(range(len(self.lengths)))
#         if self.shuffle:
#             random.shuffle(indices)
#         batch = []
#         current_max = 0
#         for idx in indices:
#             new_max = max(current_max, self.lengths[idx])
#             if new_max * (len(batch) + 1) > self.max_samples:
#
#                 yield batch
#                 batch = [idx]
#                 current_max = self.lengths[idx]
#             else:
#                 batch.append(idx)
#                 current_max = new_max
#         if batch:
#             yield batch
#
#     def __len__(self):
#         if self.len != -1:
#             return self.len
#         self.len = sum(1 for _ in self.__iter__())
#         return self.len


class DynamicBatchSampler(Sampler):
    def __init__(self, shuffle: bool, max_duration: int, lengths: list, sample_rate: int, seed: int = 42):
        super().__init__()
        self.max_samples = max_duration * sample_rate
        self.lengths = lengths
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.length_cache = {}

    def __iter__(self):

        indices = list(range(len(self.lengths)))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()
        else:
            indices = sorted(indices, key=lambda x: self.lengths[x], reverse=True)

        batch = []
        current_max = 0
        for idx in indices:
            new_max = max(current_max, self.lengths[idx])
            if new_max * (len(batch) + 1) > self.max_samples:
                # lns = [self.lengths[i] for i in batch]
                # logger.info(f'give_batch with lens: {lns}')
                # logger.info(f'max_sample_rate = {self.max_samples}   {self.max_samples // 16000}')
                # logger.info(f'batch_size= {len(batch)}')
                yield batch
                batch = [idx]
                current_max = self.lengths[idx]
            else:
                batch.append(idx)
                current_max = new_max
        if batch:

            yield batch

    def _compute_length(self):
        """Вычисляет количество батчей для текущей эпохи"""

        indices = list(range(len(self.lengths)))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()
        else:
            indices = sorted(indices, key=lambda x: self.lengths[x], reverse=True)
        batch_count = 0
        current_max = 0
        cur_len = 0

        for idx in indices:
            new_max = max(current_max, self.lengths[idx])
            if new_max * (cur_len + 1) > self.max_samples:
                batch_count += 1
                cur_len = 1
                current_max = self.lengths[idx]
            else:
                cur_len += 1
                current_max = new_max

        if cur_len > 0:
            batch_count += 1

        return batch_count

    def __len__(self):
        if self.epoch not in self.length_cache:
            self.length_cache[self.epoch] = self._compute_length()

        return self.length_cache[self.epoch]

    def set_epoch(self, epoch):
        self.epoch = epoch