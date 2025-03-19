import numpy as np
import os
import glob
import random
import soundfile
import torch
from scipy import signal


class InstanceAugment(torch.nn.Module):
    def __init__(self, musan_path, rir_path=None):
        super().__init__()
        """
        according to https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/dataLoader.py
        """

        self.noise_types = ['noise', 'speech', 'music']
        self.noise_snr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.num_noise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noise_list = {}

        augment_files = glob.glob(os.path.join(str(musan_path), '*/*/*.wav'))

        for file in augment_files:
            if file.split('/')[-3] not in self.noise_list:
                self.noise_list[file.split('/')[-3]] = []
            self.noise_list[file.split('/')[-3]].append(file)

        self.rir_files = []
        if rir_path:
            self.rir_files = glob.glob(os.path.join(str(rir_path), '*/*.wav'))
        print(self.noise_list.keys())

    def add_rev(self, audio):
        """
        Add reverberation with rirs files
        """

        if not self.rir_files:
            return audio

        assert audio.device == torch.device('cpu')
        audio_length = audio.shape[0]

        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)

        if len(rir.shape) > 1:
            rir = np.mean(rir, axis=1)

        rir = rir / np.sqrt(np.sum(rir ** 2))
        res = signal.convolve(audio.numpy(), rir, mode='full')[:audio_length]
        return torch.from_numpy(res.astype(np.float32))

    def add_noise(self, audio, noise_cat):
        """
        Add noise of a specific category to the audio
        """
        device = audio.device
        audio_length = audio.shape[0]

        clean_db = 10 * torch.log10(torch.mean(audio ** 2) + 1e-4)
        num_noise = self.num_noise[noise_cat]
        noise_list = random.sample(self.noise_list[noise_cat], random.randint(num_noise[0], num_noise[1]))
        noises = []


        for noise in noise_list:
            noise_audio, sr = soundfile.read(noise)
            if len(noise_audio) <= audio_length:
                shortage = audio_length - len(noise_audio)
                noise_audio = np.pad(noise_audio, (0, shortage), 'wrap')

            start_frame = np.int64(random.random() * (len(noise_audio) - audio_length))
            noise_audio = noise_audio[start_frame:start_frame + audio_length]
            noise_audio = torch.tensor(noise_audio, dtype=torch.float32, device=device)
            noise_db = 10 * torch.log10(torch.mean(noise_audio ** 2) + 1e-4)
            noise_snr = random.uniform(self.noise_snr[noise_cat][0], self.noise_snr[noise_cat][1])
            scaling = torch.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10))
            scaled_noise = scaling * noise_audio
            noises.append(scaled_noise)

        if noises:
            total_noise = torch.stack(noises).sum(dim=0)
            return audio + total_noise

        return audio

    def forward(self, audio, aug_type=None):
        """
        Apply augmentation to the audio

        Args:
            audio: Input audio tensor (1D tensor)
            aug_type: Augmentation type (0-5). If None, a random type is selected.

        Returns:
            Augmented audio tensor
        """
        if aug_type is None:
            aug_type = random.randint(0, 5)

        if aug_type == 0:  # Original
            return audio
        elif aug_type == 1:  # Reverberation
            return self.add_rev(audio)
        elif aug_type == 2:  # Babble
            return self.add_noise(audio, 'speech')
        elif aug_type == 3:  # Music
            return self.add_noise(audio, 'music')
        elif aug_type == 4:  # Noise
            return self.add_noise(audio, 'noise')
        elif aug_type == 5:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
            return audio
        return audio
