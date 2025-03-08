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


class AudioNormalize(nn.Module):
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
                x = x.mean(dim=0)
            else:
                x = x.squeeze(0)
        elif x.dim() == 1:
            pass
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}, expected 1D or 2D tensor")

        return x


class AdaptiveCrop1D(nn.Module):
    def __init__(self, min_sec=3.0, max_sec=40.0, sample_rate=16000):
        super().__init__()
        self.min_samples = int(min_sec * sample_rate)
        self.max_samples = int(max_sec * sample_rate)
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(3)
        self.frame_duration = 30  # ms (10, 20 или 30)
        self.frame_size = int(sample_rate * self.frame_duration / 1000)

        assert sample_rate in [8000, 16000, 32000, 48000], \
            "Unsupported sample rate"
        assert webrtcvad.valid_rate_and_frame_length(sample_rate, self.frame_size), \
            "Invalid frame size for sample rate"

    def _get_speech_segments(self, audio):
        audio = audio.astype('int16').tobytes()

        total_frames = len(audio) // (2 * self.frame_size)
        audio = audio[:total_frames * 2 * self.frame_size]

        segments = []
        current_segment = None

        for i in range(total_frames):
            start = i * 2 * self.frame_size
            end = start + 2 * self.frame_size
            frame = audio[start:end]

            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except:
                continue

            if is_speech:
                if current_segment is None:
                    current_segment = [i * self.frame_duration]
            else:
                if current_segment is not None:
                    current_segment.append(i * self.frame_duration)
                    segments.append(tuple(current_segment))
                    current_segment = None

        return segments

    def forward(self, x):
        time = x.size(0)

        if time <= self.min_samples:
            repeats = math.ceil(self.min_samples / time)
            return x.repeat(repeats)[:self.min_samples]

        audio = (x.cpu().numpy() * 32767).astype('int16')

        segments = self._get_speech_segments(audio)

        if segments:
            segments.sort(key=lambda s: s[1] - s[0], reverse=True)
            start, end = segments[0]

            cut_samples = random.randint(self.min_samples, self.max_samples)
            crop_size = min(cut_samples, end - start)
            if crop_size < cut_samples:
                start = max(0, end - cut_samples)
            else:
                offset = random.randint(0, (end - start) - cut_samples)
                start += offset

            start = max(0, start)
            end = start + cut_samples
            cropped = x[start:end]
        else:
            cut_samples = random.randint(self.min_samples, self.max_samples)
            start = max(0, (time - cut_samples) // 2)
            cropped = x[start:start + cut_samples]

        if cropped.size(0) < self.min_samples:
            repeats = math.ceil(self.min_samples / cropped.size(0))
            cropped = cropped.repeat(repeats)[:self.min_samples]
        return cropped