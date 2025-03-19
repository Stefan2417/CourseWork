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

        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(str(musan_path), '*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files = []
        if rir_path:
            self.rir_files = glob.glob(os.path.join(str(rir_path), '*/*/*.wav'))

    def add_rev(self, audio):
        """
        Add reverbation with rirs files
        """

        if not self.rir_files:
            return audio

        assert audio.device == torch.device('cpu')

        audio_length = audio.shape[1]
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)

        rir.to_device(audio.device)
        rir = np.expand_dims(rir.astype(np.float32), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :audio_length]

    def add_noise(self, audio, noisecat):
        """
        Add noise of a specific category to the audio
        """
        device = audio.device
        audio_length = audio.shape[0]

        clean_db = 10 * torch.log10(torch.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []


        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            if len(noiseaudio) <= audio_length:
                shortage = audio_length - len(noiseaudio)
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')

            start_frame = np.int64(random.random() * (len(noiseaudio) - audio_length))
            noiseaudio = noiseaudio[start_frame:start_frame + audio_length]
            noiseaudio = torch.tensor(noiseaudio, dtype=torch.float32, device=device)
            noise_db = 10 * torch.log10(torch.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            scaling = torch.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10))
            scaled_noise = scaling * noiseaudio
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
