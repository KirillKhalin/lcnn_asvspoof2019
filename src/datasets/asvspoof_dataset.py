import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset


class ASVspoof2019Dataset(Dataset):
    """ASVspoof 2019 Logical Access Dataset."""

    def __init__(self, protocol_path, dir_path, is_train=True, K=750, hop_length=160):
        self.data = []
        self.dir_path = dir_path
        self.is_train = is_train

        self.K = K
        self.hop_length = hop_length
        self.max_len = (self.K - 1) * self.hop_length  # 119840

        if os.path.exists(protocol_path):
            with open(protocol_path) as file:
                for line in file:
                    arr = line.strip().split()
                    label = 1.0 if arr[4] == "bonafide" else 0.0
                    self.data.append((arr[1], label))

    def __getitem__(self, i):
        utt_id, label = self.data[i]
        path = os.path.join(self.dir_path, utt_id + ".flac")

        wav, sr = torchaudio.load(path)
        wav = torch.squeeze(wav, dim=0)  # [Time]
        wav_len = wav.shape[0]

        if wav_len > self.max_len:
            if self.is_train:
                start = random.randint(0, wav_len - self.max_len)
            else:
                start = 0
            wav = wav[start : start + self.max_len]
        elif wav_len < self.max_len:
            rep = (self.max_len + wav_len - 1) // wav_len
            wav = wav.repeat(rep)
            wav = wav[: self.max_len]

        return {
            "audio": wav,
            "target": torch.tensor(label, dtype=torch.float32),
            "utt_id": utt_id,
        }

    def __len__(self):
        return len(self.data)
