import json
import pickle
import sys
from JSONToVector import JSONToVector

def get_tag_dicts(vector):
    tag_to_idx = {}
    idx_to_tag = {}

    idx = 0
    for tag, _, has_children, has_sibling in vector:
        if (tag, has_children, has_sibling) not in tag_to_idx:
            tag_to_idx[(tag, has_children, has_sibling)] = idx
            idx_to_tag[idx] = (tag, has_children, has_sibling)
            idx += 1

    return tag_to_idx, idx_to_tag
    

def preprocess_datasets(datasets, names):
    for dataset in datasets:
        vector = JSONToVector(*dataset)
        vector.solve()
        vector.export_json()
        serialize_dicts(get_tag_dicts(vector.res), names)


def preprocess_large_datasets():
    python_50k_eval = ('D://data//python50k_eval.json', 'D://data//python50k_eval_vector.json')
    python_100k_test = ('D://data//python100k_train.json', 'D://data//python100k_train_vector.json')

    preprocess_datasets([python_50k_eval, python_100k_test], ['python50k', 'python100k'])


def preprocess_small_datasets():
    python_1k = ('D://data//python1k.json', 'D://data//python1k_vector.json')
    preprocess_datasets([python_1k], ['python1k'])


def serialize_dicts(dicts, filenames):
    for d, fn in zip(dicts, filenames):
        with open(f'{fn}.pickle', 'wb') as f:
            pickle.dumps(d, f)


if __name__ == '__main__':
    preprocess_small_datasets()