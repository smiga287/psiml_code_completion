import json
import sys
from ASTNode import ASTNode

codes = []
bio = [False]
res = []

def dfs(index, hasparent, next):
    global codes,bio,res
    if bio[index]:
        return
    bio[index] = True
    prev= None
    val = 'EMPTY'
    if('value' in codes[index]):
        val = codes[index]['value']
    res.append([codes[index]['type'], val, hasparent, next])

    if 'children' not in codes[index]:
        return
    
    for i in codes[index]['children']:
        if(prev!=None):
            if(bio[prev]):
                continue
            dfs(prev, 'children' in codes[prev], True)
        prev=i
    dfs(prev, 'children' in codes[prev], False)
        


def solve():
    global bio, res, codes
    with open('.\data\\python100k_train.json') as json_file:
        str = json_file.read().split('\n')
        str = str[:-1]
        #codes = [json.loads(i) for i in str]
        for i in range(len(str)):
            codes = json.loads(str[i])
            bio = [False] * len(codes)

            for j in range(len(codes)):
                if bio[j]:
                    continue
                dfs( j, 'children' in codes[j], False)

            res.append(["EOF", 'EMPTY', False, False])
        

def to_txt():
    with open('.\data\\vector_of_nodes_train.json', 'w') as f:
        json.dump(res, f)


if __name__ == "__main__":
    solve()
    to_txt()