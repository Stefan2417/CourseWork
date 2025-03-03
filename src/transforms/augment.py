import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class RandomCrop1D(nn.Module):
    """
    Randomly crop 1D audio to a specified length.
    If audio is shorter than target length, it's padded with zeros.
    For instance transformation: input [time] -> output [time]
    """

    def __init__(self, max_length, sample_rate=16000):
        """
        Args:
            max_length (float): Maximum length in seconds.
            sample_rate (int): Sample rate of the audio.
        """
        super().__init__()
        self.max_samples = int(max_length * sample_rate)
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

        if time > self.max_samples:
            # Randomly select starting point for crop
            start = torch.randint(0, time - self.max_samples + 1, (1,)).item()
            x = x[start:start + self.max_samples]
        return x


class AudioNormalize(nn.Module):
    """
    Convert audio to mono and normalize amplitude.
    For instance transformation: input [channels, time] -> output [time]
    """

    def __init__(self, normalize_type="peak", target_level=None, eps=1e-8):
        """
        Args:
            normalize_type (str): Type of normalization:
                - "peak": normalize by peak amplitude
                - "rms": normalize by root mean square
                - "per_sample": normalize each sample to zero mean and unit variance
                - "fixed_level": normalize to a fixed target level
            target_level (float): Target level in dB for fixed_level normalization
            eps (float): Small value to avoid division by zero
        """
        super().__init__()
        self.normalize_type = normalize_type
        self.target_level = target_level
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (Tensor): Input audio tensor [channels, time] for instance transform
        Returns:
            x (Tensor): Mono normalized audio tensor [time]
        """
        # Check if input is multi-channel
        if x.dim() == 2:
            # Convert to mono by averaging channels if needed
            if x.size(0) > 1:
                x = x.mean(dim=0)  # Average across channels -> [time]
            else:
                x = x.squeeze(0)  # Remove channel dimension if only one channel -> [time]
        elif x.dim() == 1:
            # Already mono [time], no need to change
            pass
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}, expected 1D or 2D tensor")

        # Apply normalization based on selected type
        if self.normalize_type == "peak":
            # Normalize by peak amplitude
            peak_value = x.abs().max()
            x = x / (peak_value + self.eps)

        elif self.normalize_type == "rms":
            # Normalize by RMS value
            rms_value = torch.sqrt(torch.mean(x ** 2))
            x = x / (rms_value + self.eps)

        elif self.normalize_type == "per_sample":
            # Normalize to zero mean and unit variance
            mean = x.mean()
            std = x.std()
            x = (x - mean) / (std + self.eps)

        elif self.normalize_type == "fixed_level":
            if self.target_level is None:
                raise ValueError("target_level must be specified for fixed_level normalization")

            # Convert target level from dB to amplitude
            target_amplitude = 10 ** (self.target_level / 20)

            # Calculate current RMS
            rms_value = torch.sqrt(torch.mean(x ** 2))

            # Scale to target level
            x = x * (target_amplitude / (rms_value + self.eps))

        # Return tensor in [time] format
        return x