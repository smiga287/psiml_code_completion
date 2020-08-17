import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, data, window=50):
        self.data = data
        self.window = window

    def __getitem__(self, idx):
        return (self.data[idx : idx + self.window - 1], self.data[idx + self.window])

    def __len__(self):
        return len(self.data) - self.window


class DatasetPointer(TorchDataset):
    def __init__(self, data, values, window=50):
        # (tag_idx, val_idx, data_idx)
        self.data = torch.Tensor(
            [
                (tag_idx, val_idx, values[data_idx])
                for data_idx, (tag_idx, val_idx) in enumerate(data)
            ]
        )
        self.values = values
        self.window = window

    def __getitem__(self, idx):
        return (self.data[idx : idx + self.window - 1], self.data[idx + self.window])

    def __len__(self):
        return len(self.data) - self.window
