import json
import pickle
import sys
import random
from JSONToVector import JSONToVector


def preprocess_dataset(dataset, name):
    json_to_vec = JSONToVector(*dataset)
    json_to_vec.solve()
    json_to_vec.export_pickle()
    serialize_dicts(get_tag_dicts(json_to_vec.res), name)


def preprocess_large_datasets():
    python_50k_eval = ('python50k_eval.json', 'python50k_eval_vector.pickle')
    python_100k_test = ('python100k_train.json', 'python100k_train_vector.pickle')

    for dataset, name in [(python_50k_eval, 'python50k'), (python_100k_test, 'python100k')]:
        preprocess_dataset(dataset, name)


def preprocess_small_dataset():
    python_1k = ('python1k.json', 'python1k_vector.pickle')
    preprocess_datasets(python_1k, python_1k)


def serialize_dicts(dicts, filename):
    names = [f"{filename}_tag_to_idx", f"{filename}_idx_to_tag"]
    for d, name in zip(dicts, names):
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dumps(d, f)


def get_tag_dicts():
    with open(tag_to_idx_path, 'rb') as tti_file, open(idx_to_tag_path, 'rb') as itt_file:
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


def create_small_dataset(K = 1):
    json_to_vec = JSONToVector('python100k_train.json', 'python100k_train_vector.pickle')
    json_to_vec.solve()
    dataset = random.sample(json_to_vec.res, 1000 * K)
    dataset_exporter = JSONToVector(None, 'python1k.pickle')
    dataset_exporter.res = dataset
    dataset_exporter.export_pickle()


if __name__ == '__main__':
    create_small_dataset()