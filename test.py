import json
import warnings

import soundfile
import torch
import hydra
import soundfile as sf
import tqdm
from hydra.utils import instantiate
import torchaudio

from src.datasets.data_utils import parse_dataset_speakers
from src.utils.io_utils import read_json


# warnings.filterwarnings("ignore", category=UserWarning)

def get_speaker_embeddings(model, device, speakers_data, name, max_duration=40):
    embeddings = []
    if name not in speakers_data:
        print(name)
        return []
    for audio_path in speakers_data[name]:
        wav, sr = sf.read(audio_path)
        assert sr == 16000
        duration = len(wav) / sr
        if duration > max_duration:
            wav = wav[:int(max_duration * sr)]

        wav = torch.Tensor([wav, wav]).to(device)  # [1, T]
        wav_lengths = torch.LongTensor([wav.size(1), wav.size(1)]).to(device)

        batch = {
            "data_object" : wav,
            "lengths" : wav_lengths
        }
        with torch.no_grad():
            feats = model(batch)
            print(feats.shape)
        exit(0)
        embeddings.append({'path': audio_path, 'embedding': feats.numpy().tolist()})

    return embeddings

@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    model = instantiate(config.model).to(device)

    speakers_data = parse_dataset_speakers('VoxCeleb/vox1_dev_wav/wav', extension='wav')

    embeddings = {}
    for i in tqdm.tqdm(range(1, 2)):
        zeros = '0' * (4 - len(str(i)))
        id = f'id1{zeros}{i}'
        # print(get_speaker_embeddings(model, device, speakers_data, id))
        embeddings[id] = get_speaker_embeddings(model, device, speakers_data, id)
        # print(f"Пиковая память: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    with open('embeddings.json', 'w') as f:
        json.dump(embeddings, f, indent=2)
if __name__ == "__main__":
    main()
