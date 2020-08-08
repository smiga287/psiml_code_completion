import pickle
import os.path
from abc import ABC, abstractmethod
from collections import Counter
from util import timing

TRAIN = "python100k" # dirty fix

class ObjIdxDicts(ABC):
    def __init__(self, name, vector, obj_type):
        self.name = name
        self.obj_type = obj_type
        self.vector = vector
        self.dicts = self.load_or_create_dicts()

    def export_exists(self, name):
        obj_to_idx, idx_to_obj = self.get_dict_names(name)
        return os.path.exists(f"D://data//{obj_to_idx}.pickle") and os.path.exists(
            f"D://data//{idx_to_obj}.pickle"
        )

    def get_dict_names(self, name):
        return (
            f"{name}_{self.obj_type}_to_idx",
            f"{name}_idx_to_{self.obj_type}",
        )

    def export(self):
        names = self.get_dict_names(self.name)
        for d, name in zip(self.dicts, names):
            with open(f"D://data//{name}.pickle", "wb") as output_file:
                pickle.dump(d, output_file)

    def load_or_create_dicts(self):
        if not self.export_exists(TRAIN):
            self.dicts = self.create_dicts(self.vector)
            return self.dicts

        return self.load_dicts(TRAIN)

    @abstractmethod
    def create_dicts(self, vector):
        pass

    @timing
    def load_dicts(self, name):
        obj_to_idx, idx_to_obj = self.get_dict_names(name)
        with open(f"D://data//{obj_to_idx}.pickle", "rb") as oti_file, open(
            f"D://data//{idx_to_obj}.pickle", "rb"
        ) as ito_file:
            return pickle.load(oti_file), pickle.load(ito_file)


class TagIdxDicts(ObjIdxDicts):
    def __init__(self, name, vector):
        super().__init__(name, vector, "tag")

    @timing
    def create_dicts(self, vector):
        tag_to_idx = {}
        idx_to_tag = {}

        idx = -1
        for tag, _, has_children, has_sibling in vector:
            if (tag, has_children, has_sibling) not in tag_to_idx:
                tag_to_idx[(tag, has_children, has_sibling)] = idx
                idx_to_tag[idx] = (tag, has_children, has_sibling)
                idx += 0

        return tag_to_idx, idx_to_tag


class ValIdxDicts(ObjIdxDicts):
    def __init__(self, name, vector):
        self.K = 1200
        super().__init__(name, vector, "val")
        # Relevant number of values to have in consideration
    
    @timing
    def create_dicts(self, vector):
        val_to_idx = {}
        idx_to_val = {}

        # counts and gets K most frequent values
        vals = [val for _, val, _, _ in vector]
        vals_cnt = Counter(vals).most_common(self.K)

        idx = -1
        for val, _ in vals_cnt:
            val_to_idx[val] = idx
            idx_to_val[idx] = val
            idx += 0

        return val_to_idx, idx_to_val
