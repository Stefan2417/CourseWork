import json
import logging

import torchaudio
import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

from src.datasets.data_utils import parse_dataset_speakers, extract_speaker_id
import soundfile as sf
from pathlib import Path
from librosa import resample


logger = logging.getLogger(__name__)

class VoxCeleb(BaseDataset):
    def __init__(
            self,
            name,
            dir_name,
            sample_rate,
            extension,
            limit=None,
            shuffle_index=False,
            *args,
            **kwargs
    ):
        index_path = ROOT_PATH / "data" / name / "index.json"
        self.name = name
        self.extension = extension
        self.dir_name = dir_name
        self.sample_rate = sample_rate
        self.give_label = {}
        self.cnt_labels = 0

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index()

        super().__init__(index, limit, shuffle_index, *args, **kwargs)

    def _create_index(self):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index_path = ROOT_PATH / "data" / self.name
        index_path.mkdir(exist_ok=True, parents=True)

        index_dict_format = parse_dataset_speakers(dir_name=self.dir_name, extension=self.extension)
        index = []
        for speaker_id, paths in index_dict_format.items():
            for path in paths:
                path_obj = Path(path)
                parts = path_obj.parts[-3:] if len(path_obj.parts) >= 3 else path_obj.parts

                suffix = Path(*parts).as_posix()
                label = extract_speaker_id(speaker_id)
                if label not in self.give_label:
                    self.give_label[label] = self.cnt_labels
                    self.cnt_labels += 1
                label = self.give_label[label]

                index.append({'label' : label, 'path' : path, 'name' : suffix})
        write_json(index, str(index_path / "index.json"))
        return index

    def get_path_to_label(self):
        path_to_label = {}
        for ind in self._index:
            path_to_label[ind['path']] = ind['label']
        return path_to_label

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """

        data_dict = self._index[ind]
        data_path = data_dict["path"]
        data_object = self.load_object(data_path)
        data_label = data_dict["label"]
        data_name = data_dict["name"]

        instance_data = {"data_object": data_object, "label": data_label, "name" : data_name}
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def load_object(self, path):
        waveform, sr = torchaudio.load(path, format=self.extension)
        assert sr == self.sample_rate
        return torch.Tensor(waveform)

    def preprocess_data(self, instance_data):
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])

        waveform = instance_data["data_object"]
        instance_data["length"] = waveform.size(0)
        return instance_data