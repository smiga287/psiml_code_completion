import json
import sys

niz = None


def dfs(index, hasparent):
    pass
pass


def run():
    with open('data\python100k_train.json') as json_file:
        str = json_file.read().split('\n')
        str = str[:-1]
        niz = [json.loads(str[0])]
        dfs(niz[0], False, )



if __name__ == "__main__":
    run()