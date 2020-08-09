from JSONToVector import JSONToVector
from ObjIdxDict import TagIdxDicts, ValIdxDicts
from util import TRAIN


class DataManager:
    def __init__(self, name):
        self.name = name
        self.vectorizer = JSONToVector(self.name)
        self.data = self.vectorizer.get_data()
        self.tag_dicts = TagIdxDicts(self.name, self.data)
        self.val_dicts = ValIdxDicts(self.name, self.data)

    def get_data(self):
        return self.data

    def get_tag_dicts(self):
        return self.tag_dicts.dicts

    def get_val_dicts(self):
        return self.val_dicts.dicts

    def export(self):
        if not self.vectorizer.export_exists():
            self.vectorizer.export()
        if not self.tag_dicts.export_exists(self.name):
            self.tag_dicts.export()
        if not self.val_dicts.export_exists(self.name):
            self.val_dicts.export()


if __name__ == "__main__":
    data_manager = DataManager(TRAIN)
    tti, itt = data_manager.get_tag_dicts()
    vti, itv = data_manager.get_val_dicts()
    print()
