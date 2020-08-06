import json
import sys

class JSONToVector():
    def __init__(self, src_path: str, dest_path: str):
        self.codes = []
        self.seen = set()
        self.res = []
        self.src_path = src_path
        self.dest_path = dest_path

    def solve(self):
        json_list = self.parse_json_lines()
        for program in json_list:
            self.codes = program
            self.seen = set()

            for j, code in enumerate(self.codes):
                if j not in self.seen:
                    self.dfs(j, 'children' in code, False)

            self.res.append(['EOF', 'EMPTY', False, False])

    def dfs(self, idx, has_children, has_siblings):
        if idx in self.seen:
            return
        self.seen.add(idx)
        
        val = 'EMPTY'
        if 'value' in self.codes[idx]:
            val = self.codes[idx]['value']

        self.res.append([self.codes[idx]['type'], val, has_children, has_siblings])

        if not has_children:
            return

        prev = None
        for i in self.codes[idx]['children']:
            if prev != None and prev not in self.seen:
                self.dfs(prev, 'children' in self.codes[prev], True)
            prev = i

        self.dfs(prev, 'children' in self.codes[prev], False)
        
    def parse_json_lines(self, take=None):
        data = []
        with open(self.src_path) as json_file:
            for idx, line in enumerate(json_file):
                if take != None and idx == take:
                    break
                data.append(json.loads(line))
        return data

    def export_json(self):
        with open(self.dest_path, 'w') as dest_file:
            json.dump(self.res, dest_file)


# Regresioni test izmedju nove i stare verzije
def test_for_similarity(src, dest, b="D://data//vector_of_nodes_eval.json"):
    jtv = JSONToVector(src, dest)
    jtv.solve()
    C = 50000
    json_data = json.dumps(jtv.res[:2000])[:C]
    with open(b) as original:
        s = original.read(C)
        for i in range(C):
            if s[i] != json_data[i]:
                print(s[i - 20:i + 20])
                print(json_data[i - 20:i + 20])
                break



    