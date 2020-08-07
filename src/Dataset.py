from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, data, window=50):
        self.data = data
        self.window = window

    def __getitem__(self, idx):
        return (self.data[idx : idx + self.window-1], self.data[idx+self.window])

    def __len__(self):
        return len(self.data) - self.window
