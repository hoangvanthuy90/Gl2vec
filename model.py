import numpy as np
import networkx as nx
from typing import List
from base import Base
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import WeisfeilerLehmanHashing


class GL2Vec(Base):

    def __init__(
        self,
        wl_iterations: int = 2,
        dimensions: int = 128,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
    ):

        self.wl_iterations = wl_iterations
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features

    def create_line_graph(self, graph):
        graph = nx.line_graph(graph)
        node_mapper = {node: i for i, node in enumerate(graph.nodes())}
        edges = [[node_mapper[edge[0]], node_mapper[edge[1]]] for edge in graph.edges()]
        line_graph = nx.from_edgelist(edges)
        return line_graph

    def train(self, graphs: List[nx.classes.graph.Graph]):
        self.set_seed()
        graphs = self.check_graphs(graphs)
        graphs = [self.create_line_graph(graph) for graph in graphs]
        documents = [
            WeisfeilerLehmanHashing(
                graph, self.wl_iterations, False, self.erase_base_features
            )
            for graph in graphs
        ]
        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        self.model = Doc2Vec(
            documents,
            vector_size=self.dimensions,
            window=0,
            min_count=self.min_count,
            dm=0,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )

        self._embedding = [self.model.docvecs[str(i)] for i, _ in enumerate(documents)]

    def get_embedding(self) -> np.array:
        return np.array(self._embedding)

    def infer(self, graphs) -> np.array:
        self._set_seed()
        graphs = self._check_graphs(graphs)
        graphs = [self._create_line_graph(graph) for graph in graphs]
        documents = [
            WeisfeilerLehmanHashing(
                graph, self.wl_iterations, False, self.erase_base_features
            )
            for graph in graphs
        ]

        documents = [doc.get_graph_features() for _, doc in enumerate(documents)]

        embedding = np.array(
            [
                self.model.infer_vector(
                    doc, alpha=self.learning_rate, min_alpha=0.00001, epochs=self.epochs
                )
                for doc in documents
            ]
        )

        return embedding