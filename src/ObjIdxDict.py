import pickle
import os.path
from abc import ABC, abstractmethod
from collections import Counter
from util import timing, DATA_ROOT, TRAIN, SMALL


class ObjIdxDicts(ABC):
    def __init__(self, name, vector, obj_type):
        self.name = name
        self.obj_type = obj_type
        self.vector = vector
        if self.export_exists(TRAIN):
            self.dicts = self.load_dicts(TRAIN)
        else:
            self.dicts = self.create_dicts(self.vector)

    def _get_dict_names(self, name):
        return (
            f"{name}_{self.obj_type}_to_idx",
            f"{name}_idx_to_{self.obj_type}",
        )

    def _export_path(self, name, obj_type):
        return f"{DATA_ROOT}{name}_{obj_type}_dicts.pickle"

    def export_exists(self, name):
        return os.path.exists(self._export_path(name, self.obj_type))

    def export(self):
        obj_to_idx_name, idx_to_obj_name = self._get_dict_names(self.name)
        obj_to_idx, idx_to_obj = self.dicts
        with open(self._export_path(self.name, self.obj_type), "wb") as f:
            save = {f"{obj_to_idx_name}": obj_to_idx, f"{idx_to_obj_name}": idx_to_obj}
            pickle.dump(save, f, protocol=4)

    @timing
    def load_dicts(self, name):
        obj_to_idx_name, idx_to_obj_name = self._get_dict_names(name)
        with open(self._export_path(name, self.obj_type), "rb") as f:
            save = pickle.load(f)
            return save[obj_to_idx_name], save[idx_to_obj_name]

    @abstractmethod
    def create_dicts(self, vector):
        pass


class TagIdxDicts(ObjIdxDicts):
    def __init__(self, name, vector):
        super().__init__(name, vector, "tag")

    @timing
    def create_dicts(self, vector):
        tag_to_idx = {}
        idx_to_tag = {}

        idx = 0
        for tag, _, has_children, has_sibling in vector:
            if (tag, has_children, has_sibling) not in tag_to_idx:
                tag_to_idx[(tag, has_children, has_sibling)] = idx
                idx_to_tag[idx] = (tag, has_children, has_sibling)
                idx += 1

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

        idx = 0
        for val, _ in vals_cnt:
            val_to_idx[val] = idx
            idx_to_val[idx] = val
            idx += 1

        return val_to_idx, idx_to_val


if __name__ == "__main__":
    from JSONToVector import JSONToVector

    tag = TagIdxDicts(SMALL, JSONToVector(SMALL).get_data())
    tag.export()

    val = ValIdxDicts(SMALL, JSONToVector(SMALL).get_data())
    val.export()
