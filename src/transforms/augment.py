import torch
import torchaudio
from pathlib import Path
import random


class MUSANAugment(torch.nn.Module):
    def __init__(self, musan_path, skip=False, sample_rate=16000, min_snr_db=10, max_snr_db=20):
        super().__init__()
        self.skip = skip
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.musan_path = Path(musan_path)

        self.noise_files = self._get_files("noise")
        self.speech_files = self._get_files("speech")
        self.music_files = self._get_files("music")

        self._validate_dataset()

    def _get_files(self, category):
        path = self.musan_path / category
        return sorted(path.rglob("*.wav")) if path.exists() else []

    def _validate_dataset(self):
        if not all([self.noise_files, self.speech_files, self.music_files]):
            raise ValueError(f"Invalid MUSAN dataset at {self.musan_path}")

    def _load_noise(self, file_path, target_length, device):
        waveform, sr = torchaudio.load(file_path, channels_first=False)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform.T, sr, self.sample_rate).T

        if waveform.shape[1] > 1:  # [samples, channels]
            waveform = waveform.mean(dim=1, keepdim=True)

        waveform = waveform.to(device)

        if waveform.size(0) < target_length:
            repeats = (target_length // waveform.size(0)) + 1
            waveform = waveform.repeat(repeats, 1)

        return waveform[:target_length, 0]  # [time]

    def forward(self, audio):
        """
        Args:
            audio: Tensor shape [batch, time]
        Returns:
            Augmented audio: Tensor shape [batch, time]
        """
        if self.skip:
            return audio
        batch_size, target_length = audio.shape
        device = audio.device

        aug_type = random.choice(["noise", "speech", "music"])
        file_path = random.choice(getattr(self, f"{aug_type}_files"))

        noise = self._load_noise(file_path, target_length, device)  # [time]

        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        audio_power = audio.pow(2).mean(dim=-1)  # [batch]
        noise_power = noise.pow(2).mean()

        scale = torch.sqrt(audio_power / (noise_power * (10 ** (snr_db / 10)) + 1e-8))
        scaled_noise = scale.view(-1, 1) * noise  # [batch, time]

        return audio + scaled_noise

    def __repr__(self):
        return (f"MonoMUSANAugment(noise={len(self.noise_files)}, "
                f"speech={len(self.speech_files)}, music={len(self.music_files)})")
