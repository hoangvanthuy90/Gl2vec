import io
import json
import numpy as np
import pandas as pd
import networkx as nx
from typing import List
from six.moves import urllib

class GraphLoader(object):

    def __init__(self, dataset: str = "reddit10k"):
        self.dataset = dataset
        self.base_url = "https://github.com/benedekrozemberczki/karateclub/raw/master/dataset/graph_level/"

    def _pandas_reader(self, bytes):
        tab = pd.read_csv(
            io.BytesIO(bytes), encoding="utf8", sep=",", dtype={"switch": np.int32}
        )
        return tab

    def _dataset_reader(self, end):
        path = self.base_url + self.dataset + "/" + end
        data = urllib.request.urlopen(path).read()
        return data

    def get_graphs(self) -> List[nx.classes.graph.Graph]:
        graphs = self._dataset_reader("graphs.json")
        graphs = json.loads(graphs.decode())
        graphs = [nx.from_edgelist(graphs[str(i)]) for i in range(len(graphs))]
        return graphs

    def get_target(self) -> np.array:
        data = self._dataset_reader("target.csv")
        data_tab = self._pandas_reader(data)
        target = np.array(data_tab["target"])
        return target
