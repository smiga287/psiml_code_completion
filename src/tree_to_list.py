import json
import sys
from ASTNode import ASTNode

codes = []
bio = [False]*100
res = []

def dfs(code, index, hasparent, next):
    global codes,bio,res
    if bio[index]:
        return
    bio[index] = True
    prev= None
    val = 'EMPTY'
    if('value' in codes[code][index]):
        val = codes[code][index]['value']
    res.append([codes[code][index]['type'], val, hasparent, next])

    if 'children' not in codes[code][index]:
        return
    
    for i in codes[code][index]['children']:
        if(prev!=None):
            if(bio[prev]):
                continue
            dfs(code, prev, len(codes[code][prev]['children'])>0, True)
        prev=i
        dfs(code, prev, 'children' in codes[code][prev], False)
        


def solve():
    global bio, res, codes
    with open('.\data\\temp.json') as json_file:
        str = json_file.read().split('\n')
        str = str[:-1]
        codes = [json.loads(str[0])]
        for i in range(len(codes)):

            bio = [False] * len(codes[i])

            for j in range(len(codes[i])):
                if bio[j]:
                    continue
                dfs(i, j, 'children' in codes[i][j], False)

            res.append(["EOF", 'EMPTY', False, False])
        

def to_txt():
    with open('.\data\\vector_of_nodes.json', 'w') as f:
        json.dump(res, f)


if __name__ == "__main__":
    solve()
    to_txt()