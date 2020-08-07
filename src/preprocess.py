import json
import pickle
import sys
import random
import os.path
from collections import Counter
from JSONToVector import JSONToVector


def preprocess_dataset(dataset, name):
    json_to_vec = JSONToVector(*dataset)
    json_to_vec.solve()
    if not vector_export_exists(name):
        json_to_vec.export_pickle()
    if not dict_export_exists(name, 'tag'):
        serialize_dicts(create_tag_dicts(json_to_vec.res), name, 'tag')
    if not dict_export_exists(name, 'val'):
        serialize_dicts(create_val_dicts(json_to_vec.res), name, 'val')



def vector_export_exists(name):
    return os.path.exists(f'D://data//{name}_vector.pickle')


def dict_export_exists(name, obj_type):
    obj_to_idx, idx_to_obj = get_dict_names(name, obj_type)
    return os.path.exists(f'D://data//{obj_to_idx}.pickle') and os.path.exists(f'D://data//{idx_to_obj}.pickle') 


def preprocess_train_datasets():
    python_100k_train = ("python100k_train.json", "python100k_vector.pickle")
    preprocess_dataset(python_100k_train, 'python100k')


def preprocess_eval_datasets():
    python_50k_eval = ("python50k_eval.json", "python50k_vector.pickle")
    preprocess_dataset(python_50k_eval, 'python50k')


def preprocess_small_dataset():
    python_1k = ("python1k.json", "python1k_vector.pickle")
    preprocess_dataset(python_1k, "python1k")


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


def get_tag_dicts(filename="python100k"):
    tag_to_idx_filename, idx_to_tag_filename = map(
        lambda name: f"{name}.pickle", get_dict_names(filename, 'tag')
    )
    with open(f"D://data//{tag_to_idx_filename}", "rb") as tti_file, open(
        f"D://data//{idx_to_tag_filename}", "rb"
    ) as itt_file:
        return pickle.load(tti_file), pickle.load(itt_file)


def create_val_dicts(vector):
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


def calc_num_of_most_freq_vals(vector):
        return 1200


def get_val_dicts(filename="python100k"):
    val_to_idx_filename, idx_to_val_filename = map(
        lambda name: f"{name}.pickle", get_dict_names(filename, 'val')
    )
    with open(f"D://data//{val_to_idx_filename}", "rb") as vti_file, open(
        f"D://data//{idx_to_val_filename}", "rb"
    ) as itv_file:
        return pickle.load(vti_file), pickle.load(itv_file)


def get_dict_names(filename, obj_type):
    return f"{filename}_{obj_type}_to_idx", f"{filename}_idx_to_{obj_type}"


def serialize_dicts(dicts, filename, obj_type):
    names = get_dict_names(filename, obj_type)
    for d, name in zip(dicts, names):
        with open(f"D://data//{name}.pickle", "wb") as f:
            pickle.dump(d, f)


def create_small_dataset():
    with open("D://data//python100k_train_vector.pickle", "rb") as large_file, open(
        "D://data//python1k.pickle", "wb"
    ) as small_file:
        large = pickle.load(large_file)
        small = large[:300000]
        pickle.dump(small, small_file)


def get_small_dataset():
    with open("D://data//python1k_vector.pickle", "rb") as f:
        return pickle.load(f)


def get_train_dataset():
    with open("D://data//python100k_train_vector.pickle", "rb") as f:
        return pickle.load(f)


def get_eval_dataset():
    with open("D://data//python50k_eval_vector.pickle", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    a, b = get_val_dicts()
    print()
