import json
import sys

READ_FROM1 = 'D://data//vector_of_nodes_train.json'
READ_FROM2 = 'D://data//vector_of_nodes_eval.json'
WRITE_TO1 = 'D://data//tag_to_idx.json'
WRITE_TO2 = 'D://data//idx_to_tag.json'

#todo: number of diferent 'types' in paper and data is different????????????????????????????

def proces():
    tag_to_idx = {}
    val_to_idx = {}
    idx_to_tag = {}
    idx_to_val = {}
    with open(READ_FROM1) as f:
        vector = json.loads(f.read())
        #print(len(vector))
        for node in vector:
            if (node[0], node[2], node[3]) not in tag_to_idx:
                tag_to_idx[(node[0], node[2], node[3])] = len(tag_to_idx)
                idx_to_tag[len(tag_to_idx)-1] = (node[0], node[2], node[3])
        #for x in tag_to_idx:
    print(len(tag_to_idx))
    with open(READ_FROM2) as f:
        vector = json.loads(f.read())
        
        for node in vector:
            if (node[0], node[2], node[3]) not in tag_to_idx:
                tag_to_idx[(node[0], node[2], node[3])] = len(tag_to_idx)
                idx_to_tag[len(tag_to_idx)-1] = (node[0], node[2], node[3])
        
    with open(WRITE_TO2, 'w') as f:
        json.dump(idx_to_tag, f)

    print(len(tag_to_idx))


if __name__ == "__main__":
    proces()