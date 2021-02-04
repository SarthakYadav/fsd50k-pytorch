import os
import tqdm
import glob
import json
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from torch.utils.data import Dataset
from src.data.utils import load_audio
from src.data.audio_parser import AudioParser


class FSD50kEvalDataset(Dataset):
    def __init__(self, manifest_path: str, labels_map: str,
                 audio_config: dict,
                 labels_delimiter: Optional[str] = ",",
                 transform: Optional = None) -> None:
        super(FSD50kEvalDataset, self).__init__()
        assert os.path.isfile(labels_map)
        assert os.path.splitext(labels_map)[-1] == ".json"
        assert audio_config is not None
        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)

        self.len = None
        self.labels_delim = labels_delimiter
        df = pd.read_csv(manifest_path)
        self.files = df['files'].values
        self.labels = df['labels'].values
        self.exts = df['ext'].values
        self.unique_exts = np.unique(self.exts)

        assert len(self.files) == len(self.labels) == len(self.exts)
        self.len = len(self.unique_exts)
        print("LENGTH OF VAL SET:", self.len)
        self.sr = audio_config.get("sample_rate", "22050")
        self.n_fft = audio_config.get("n_fft", 511)
        win_len = audio_config.get("win_len", None)
        if not win_len:
            self.win_len = self.n_fft
        else:
            self.win_len = win_len
        hop_len = audio_config.get("hop_len", None)
        if not hop_len:
            self.hop_len = self.n_fft // 2
        else:
            self.hop_len = hop_len

        self.normalize = audio_config.get("normalize", False)
        self.min_duration = audio_config.get("min_duration", None)

        feature = audio_config.get("feature", "spectrogram")
        self.spec_parser = AudioParser(n_fft=self.n_fft, win_length=self.win_len,
                                       hop_length=self.hop_len, feature=feature)
        self.transform = transform

    def __get_audio__(self, f):
        audio = load_audio(f, self.sr, self.min_duration)
        return audio

    def __get_feature__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp = self.spec_parser(audio)
        return real, comp

    def __get_item_helper__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.files[index]
        lbls = self.labels[index]
        label_tensor = self.__parse_labels__(lbls)
        preprocessed_audio = self.__get_audio__(f)
        real, comp = self.__get_feature__(preprocessed_audio)
        if self.transform is not None:
            real = self.transform(real)
        return real, comp, label_tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt_ext = self.unique_exts[index]
        idxs = np.where(self.exts == tgt_ext)[0]

        tensors = []
        label_tensors = []
        for idx in idxs:
            real, comp, label_tensor = self.__get_item_helper__(idx)
            tensors.append(real.unsqueeze(0).unsqueeze(0))
            label_tensors.append(label_tensor.unsqueeze(0))

        tensors = torch.cat(tensors)
        return tensors, label_tensors[0]

    def __parse_labels__(self, lbls: str) -> torch.Tensor:
        label_tensor = torch.zeros(len(self.labels_map)).float()
        for lbl in lbls.split(self.labels_delim):
            label_tensor[self.labels_map[lbl]] = 1

        return label_tensor

    def __len__(self):
        return self.len


def _collate_fn_eval(batch):
    return batch[0][0], batch[0][1]
