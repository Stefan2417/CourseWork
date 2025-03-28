import logging

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

from src.datasets.data_utils import parse_dataset_speakers, extract_speaker_id
from pathlib import Path


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
        self.dir_name = dir_name
        self.extension = extension
        self.sample_rate = sample_rate
        self.give_label = {}
        self.cnt_labels = 0

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index()

        super().__init__(index=index, sample_rate=sample_rate, limit=limit, shuffle_index=shuffle_index, *args, **kwargs)

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
