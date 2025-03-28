import os
import random

import torchaudio
import tqdm
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

from src.datasets.data_utils import parse_dataset_speakers, extract_speaker_id
from pathlib import Path

import re
def extract_gender(filename):
    match = re.search(r'(\d+)([mf])', filename)
    if match:
        gender_code = match.group(2)
        return 'male' if gender_code == 'm' else 'female'
    return None


def get_splits(index_dict_format, give_label):

    index = []
    for speaker_id, paths in index_dict_format.items():
        speaker_id = f'id{speaker_id}'
        for path in paths:
            path_obj = Path(path)
            parts = path_obj.parts[-3:] if len(path_obj.parts) >= 3 else path_obj.parts

            suffix = Path(*parts).as_posix()
            label = extract_speaker_id(speaker_id)
            if label not in give_label:
                give_label[label] = len(give_label)
            label = give_label[label]

            index.append({'label': label, 'path': path, 'name': suffix})

    was = set()
    bad_paths = []
    good_paths = []

    for speaker_id1, paths_1 in index_dict_format.items():
        for speaker_id2, paths_2 in index_dict_format.items():
            for paths1 in paths_1:
                for paths2 in paths_2:
                    if paths1 == paths2:
                        continue
                    if (paths1, paths2) in was or (paths2, paths1) in was:
                        continue
                    was.add((paths1, paths2))
                    ok = int(speaker_id1 == speaker_id2)
                    if ok:
                        good_paths.append(f'{int(speaker_id1 == speaker_id2)} {Path(*Path(paths1).parts[-3:])} {Path(*Path(paths2).parts[-3:])}\n')
                    else:
                        bad_paths.append(f'{int(speaker_id1 == speaker_id2)} {Path(*Path(paths1).parts[-3:])} {Path(*Path(paths2).parts[-3:])}\n')

    return index, good_paths, bad_paths

if __name__ == "__main__":
    index_path = ROOT_PATH / "ZULU"
    index_dict_format = parse_dataset_speakers(dir_name='/home/stefan/Documents/CourseWork/ZULU/audio', extension="wav")
    deleted = []
    for speaker_id in index_dict_format:
        cur = index_dict_format[speaker_id]
        good = []
        for path in cur:
            wav, sr = torchaudio.load(path)
            if wav.shape[1] < 5 * sr:
                continue
            good.append(path)
        if len(good) < 80:
            deleted.append(speaker_id)
            continue
        random.shuffle(good)
        index_dict_format[speaker_id] = good[:80]
    for i in deleted:
        index_dict_format.pop(i)

    gender = {}
    for speaker_id, paths in index_dict_format.items():
        path = paths[0]
        gender[speaker_id] = extract_gender(path)
    males = dict(f for f in list(index_dict_format.items()) if gender[f[0]] == 'male')
    females = dict(f for f in list(index_dict_format.items()) if gender[f[0]] == 'female')

    males = dict([f for f in males.items()][:20])
    females = dict([f for f in females.items()][:20])

    give_labels = {}
    indexm, good_pathsm, bad_pathsm = get_splits(males, give_labels)
    indexf, good_pathsf, bad_pathsf = get_splits(females, give_labels)


    index = indexm + indexf
    good = good_pathsm + good_pathsf
    bad = bad_pathsm + bad_pathsf

    bad = random.sample(bad, len(good))

    split = good + bad

    write_json(index, index_path/'index.json')
    with open('test_split_zulu.txt', 'w') as f:
        for s in split:
            f.write(s)





