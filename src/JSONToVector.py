import os
import json
import pickle


class JSONToVector:
    def __init__(self, name: str):
        self.codes = []
        self.seen = set()
        self.res = []
        self.src_path = f"D://data//{name}.json"
        self.dest_path = f"D://data//{name}_vector.pickle"
        self.solve()

    def get_data(self):
        return self.res

    def solve(self):
        json_list = self.parse_json_lines()
        for program in json_list:
            self.codes = program
            self.seen = set()

            for j, code in enumerate(self.codes):
                if j not in self.seen:
                    self.dfs(j, "children" in code, False)

            self.res.append(["EOF", "EMPTY", False, False])

    def dfs(self, idx, has_children, has_siblings):
        if idx in self.seen:
            return
        self.seen.add(idx)

        val = "EMPTY"
        if "value" in self.codes[idx]:
            val = self.codes[idx]["value"]

        self.res.append([self.codes[idx]["type"], val, has_children, has_siblings])

        if not has_children:
            return

        prev = None
        for i in self.codes[idx]["children"]:
            if prev is not None and prev not in self.seen:
                self.dfs(prev, "children" in self.codes[prev], True)
            prev = i

        self.dfs(prev, "children" in self.codes[prev], False)

    def parse_json_lines(self, take=None):
        data = []
        with open(self.src_path) as json_file:
            for idx, line in enumerate(json_file):
                if take is not None and idx == take:
                    break
                data.append(json.loads(line))
        return data

    def export(self):
        with open(self.dest_path, "wb") as dest_file:
            pickle.dump(self.res, dest_file)

    def export_exists(self):
        return os.path.exists(self.dest_path)
