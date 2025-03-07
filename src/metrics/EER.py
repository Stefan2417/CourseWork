import logging
import os

import torch
import numpy as np
from sklearn.metrics import roc_curve
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from src.metrics.base_metric import BaseMetric
logger = logging.getLogger(__name__)

class EERMetric(BaseMetric):

    def __init__(self,
                 name: str,
                 pairs_path: str,
                 device: torch.device):
        super().__init__(name=name)
        self.pairs = self._load_pairs(Path(pairs_path))
        self.can_cached = 0
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.embeddings_cache: Dict[str, torch.Tensor] = {}

    def _load_pairs(self, path: Path) -> List[Tuple[int, str, str]]:
        pairs = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    label, path1, path2 = parts
                    pairs.append((int(label), path1, path2))
        return pairs

    def _cache_embeddings(self, directory=None, **kwargs) -> None:
        if 'batch_embeddings' in kwargs:
            batch = kwargs['batch_embeddings']
            batch_size = batch["embeddings"].shape[0]
            for i in range(batch_size):
                self.can_cached += 1
                # clone because of
                # https://github.com/pytorch/pytorch/issues/1995
                embedding = batch["embeddings"][i].clone()
                name = batch["names"][i]
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                self.embeddings_cache[name] = embedding
        else:
            for filename in os.listdir(directory):
                if filename.endswith(".pth"):
                    file_path = os.path.join(directory, filename)
                    try:
                        data = torch.load(file_path)
                        key = data['name']  # Используем имя файла без расширения как ключ
                        embedding = torch.nn.functional.normalize(data['embedding'], p=2, dim=0)
                        self.embeddings_cache[key] = embedding
                    except Exception as e:
                        logger.info(f"Ошибка при загрузке {filename}: {e}")


    def __call__(self, directory=None, **kwargs) -> float:
        """
        Args:
        """

        if len(self.embeddings_cache) == 0 or 'batch_embeddings' in kwargs:
            self._cache_embeddings(directory, **kwargs)

        if 'batch_embeddings' in kwargs:
            return -1

        similarities, labels = [], []
        for label, name1, name2 in self.pairs:
            emb1 = self.embeddings_cache.get(name1)
            emb2 = self.embeddings_cache.get(name2)

            if emb1 is None:
                logger.info(f'{label}, {name1}, {name2}, does not exist {name1}')
                continue
                # raise ValueError(f'embedding for {name1} does not exist')
            if emb2 is None:
                logger.info(f'{label}, {name1}, {name2}, does not exist {name2}')
                continue
                # raise ValueError(f'embedding for {name2} does not exist')
            emb1.to(self.device)
            emb2.to(self.device)

            cos_sim = torch.nn.functional.cosine_similarity(
                emb1, emb2, dim=0).item()
            similarities.append(cos_sim)
            labels.append(label)

        if not labels:
            raise ValueError("No valid pairs found for EER calculation")

        fpr, tpr, _ = roc_curve(labels, similarities)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

        # plt.hist([s for s, l in zip(similarities, labels) if l == 1], alpha=0.5, label='Genuine')
        # plt.hist([s for s, l in zip(similarities, labels) if l == 0], alpha=0.5, label='Impostor')
        # plt.legend()
        # plt.savefig("eer_distribution.png")  # Сохранит в рабочую директорию
        # plt.close()  # Освободит память

        return eer

    def reset(self):
        self.embeddings_cache.clear()