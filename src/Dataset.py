from JSONToVector import JSONToVector
from torch.utils.data import Dataset as TorchDataset
from ObjIdxDict import TagIdxDicts, ValIdxDicts


TRAIN = "python100k"
EVAL = "python100k_eval"
TEST = "python50k"
SMALL = "python1k"


class Dataset(TorchDataset):
    def __init__(self, name, window=50):
        self.name = name
        self.window = window
        self.vectorizer = JSONToVector(self.name)
        self.data = self.vectorizer.get_data()
        self.tag_dicts = TagIdxDicts(self.name, self.data)
        self.val_dicts = ValIdxDicts(self.name, self.data)

    def __getitem__(self, index):
        return self.data[index : index + self.window]

    def __len__(self):
        return len(self.data) - self.window

    def export(self):
        if not self.vectorizer.export_exists():
            self.vectorizer.export()
        if not self.tag_dicts.export_exists():
            self.tag_dicts.export()
        if not self.val_dicts.export_exists():
            self.val_dicts.export()
