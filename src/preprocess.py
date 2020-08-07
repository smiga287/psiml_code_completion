import json
import pickle
import sys
import random
from collections import Counter
from JSONToVector import JSONToVector


def preprocess_dataset(dataset, name):
    json_to_vec = JSONToVector(*dataset)
    json_to_vec.solve()
    json_to_vec.export_pickle()
    serialize_dicts(get_tag_dicts(), name)


def preprocess_large_datasets():
    python_50k_eval = ("python50k_eval.json", "python50k_eval_vector.pickle")
    python_100k_test = ("python100k_train.json", "python100k_train_vector.pickle")

    for dataset, name in [
        (python_50k_eval, "python50k"),
        (python_100k_test, "python100k"),
    ]:
        preprocess_dataset(dataset, name)


def preprocess_small_dataset():
    python_1k = ("python1k.json", "python1k_vector.pickle")
    preprocess_dataset(python_1k, "python_1k")


def get_dict_names(filename):
    return f"{filename}_tag_to_idx", f"{filename}_idx_to_tag"


def serialize_dicts(dicts, filename):
    names = get_dict_names(filename)
    for d, name in zip(dicts, names):
        with open(f"D://data//{name}.pickle", "wb") as f:
            pickle.dump(d, f)


def get_tag_dicts(filename="python100k"):
    tag_to_idx_path, idx_to_tag_path = map(
        lambda name: f"{name}.pickle", get_dict_names(filename)
    )
    with open(f"D://data//{tag_to_idx_path}", "rb") as tti_file, open(
        f"D://data//{idx_to_tag_path}", "rb"
    ) as itt_file:
        return pickle.load(tti_file), pickle.load(itt_file)


def create_tag_dicts(vector):
    tag_to_idx = {}
    idx_to_tag = {}

    idx = 0
    for tag, _, has_children, has_sibling in vector:
        if (tag, has_children, has_sibling) not in tag_to_idx:
            tag_to_idx[(tag, has_children, has_sibling)] = idx
            idx_to_tag[idx] = (tag, has_children, has_sibling)
            idx += 1

    return tag_to_idx, idx_to_tag


def calc_num_of_most_freq_vals(vector):
    return 1200


def create_value_dicts(vector):
    val_to_idx = {}
    idx_to_val = {}

    K = calc_num_of_most_freq_vals(vector)

    # counts and gets K most frequent values
    vals = [val for _, val, _, _ in vector]
    vals_cnt = Counter(vals).most_common(K)

    idx = 0
    for val, _ in vals_cnt:
        val_to_idx[val] = idx
        idx_to_val[idx] = val
        idx += 1

    return val_to_idx, idx_to_val


def create_small_dataset():
    with open("D://data//python100k_train_vector.pickle", "rb") as large_file, open(
        "D://data//python1k.pickle", "wb"
    ) as small_file:
        large = pickle.load(large_file)
        small = large[:300000]
        pickle.dump(small, small_file)


def get_small_dataset():
    with open("./data/python1k.pickle", "rb") as f:
        return pickle.load(f)


def get_train_dataset():
    with open("D://data//python100k_train_vector.pickle", "rb") as f:
        return pickle.load(f)


def get_eval_dataset():
    with open("D://data//python50k_eval_vector.pickle", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    vti, itv = create_value_dicts(get_small_dataset())
    print()
