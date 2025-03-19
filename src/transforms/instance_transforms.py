import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import random
import math
import webrtcvad


class RandomCrop1D(nn.Module):
    """
    Randomly crop 1D audio to a specified length.
    If audio is shorter than target length, it's padded with zeros.
    For instance transformation: input [time] -> output [time]
    """

    def __init__(self, min_sec=4.0, max_sec=20.0, sample_rate=16000):
        """
        Args:
            max_length (float): Maximum length in seconds.
            sample_rate (int): Sample rate of the audio.
        """
        super().__init__()
        self.min_samples = int(min_sec * sample_rate)
        self.max_samples = int(max_sec * sample_rate)
        self.sample_rate = sample_rate

    def forward(self, x):
        """
        Args:
            x (Tensor): Input audio tensor [time] for instance transform
        Returns:
            x (Tensor): Randomly cropped audio tensor [time]
        """
        # x is expected to be [time] after AudioNormalize
        if x.dim() != 1:
            raise ValueError(f"Expected 1D tensor, got {x.dim()}D tensor")

        time = x.size(0)

        cut_samples = random.randint(self.min_samples, self.max_samples)

        cropped = x
        if time > cut_samples:
            # Randomly select starting point for crop
            start = random.randint(0, time - cut_samples)
            cropped = x[start:start + self.max_samples]
        elif time < self.min_samples:
            repeats = math.ceil(self.min_samples / time)
            cropped = x.repeat(repeats)[:self.min_samples]
        return cropped


class AudioSizeNormalize(nn.Module):
    """
    Convert audio to mono and normalize amplitude.
    For instance transformation: input [channels, time] -> output [time]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input audio tensor [channels, time] for instance transform
        Returns:
            x (Tensor): Mono normalized audio tensor [time]
        """
        if x.dim() == 2:
            if x.size(0) > 1:
                x = x[0]
            else:
                x = x.squeeze(0)
        elif x.dim() == 1:
            pass
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}, expected 1D or 2D tensor")

        return x