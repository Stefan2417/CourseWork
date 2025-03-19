import torch
import torchaudio
from pathlib import Path
import random
import pyarrow as pa


class MUSANAugment(torch.nn.Module):
    def __init__(self, musan_path, sample_rate=16000, min_snr_db=10, max_snr_db=20, cache_to_memory=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.musan_path = Path(musan_path)
        self.cache_to_memory = cache_to_memory

        self.noise_paths = self._get_files("noise")
        self.speech_paths = self._get_files("speech")
        self.music_paths = self._get_files("music")

        self._validate_dataset()

        if self.cache_to_memory:
            self.noise_cache = self._preload_category(self.noise_paths)
            self.speech_cache = self._preload_category(self.speech_paths)
            self.music_cache = self._preload_category(self.music_paths)

    def _get_files(self, category):
        path = self.musan_path / category
        return sorted(path.rglob("*.wav")) if path.exists() else []

    def _preload_category(self, file_paths):
        """Preload all audio files in a category into tensors for sharing across processes"""
        pa_arrays = []

        for file_path in file_paths:
            waveform, sr = torchaudio.load(file_path, channels_first=False)
            assert sr == self.sample_rate
            assert waveform.shape[1] == 1
            waveform_np = waveform.squeeze(1).numpy()
            pa_arrays.append(pa.array(waveform_np))

        return pa_arrays

    def _validate_dataset(self):
        if not all([self.noise_paths, self.speech_paths, self.music_paths]):
            raise ValueError(f"Invalid MUSAN dataset at {self.musan_path}")

    def _get_noise(self, category, target_length, device):
        """Get a random noise sample from the specified category"""
        if self.cache_to_memory:
            cache = getattr(self, f"{category}_cache")
            pa_array = random.choice(cache)
            waveform = torch.from_numpy(pa_array.to_numpy()).float()
        else:
            paths = getattr(self, f"{category}_paths")
            file_path = random.choice(paths)
            waveform, sr = torchaudio.load(file_path, channels_first=False)
            assert sr == self.sample_rate
            assert waveform.shape[1] == 1
            waveform = waveform.squeeze(1)

        if waveform.size(0) < target_length:
            repeats = (target_length // waveform.size(0)) + 1
            waveform = waveform.repeat(repeats)

        return waveform[:target_length].to(device)

    def forward(self, audio):
        """
        Args:
            audio: Tensor shape [batch, time]
        Returns:
            Augmented audio: Tensor shape [batch, time]
        """
        batch_size, target_length = audio.shape
        device = audio.device

        aug_type = random.choice(["noise", "speech", "music"])
        noise = self._get_noise(aug_type, target_length, device)  # [time]

        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        audio_power = audio.pow(2).mean(dim=-1)  # [batch]
        noise_power = noise.pow(2).mean()

        scale = torch.sqrt(audio_power / (noise_power * (10 ** (snr_db / 10)) + 1e-8))
        scaled_noise = scale.view(-1, 1) * noise  # [batch, time]

        return audio + scaled_noise

    def __repr__(self):
        if self.cache_to_memory:
            return (f"MUSANAugment(noise={len(self.noise_cache)}, "
                    f"speech={len(self.speech_cache)}, music={len(self.music_cache)}, cached=True)")
        else:
            return (f"MUSANAugment(noise={len(self.noise_paths)}, "
                    f"speech={len(self.speech_paths)}, music={len(self.music_paths)}, cached=False)")
