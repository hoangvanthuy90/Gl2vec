import networkx as nx
from model import GL2Vec
from dataloader import GraphLoader

reader = GraphLoader()

graphs = reader.get_graphs()

model = GL2Vec()

model.train(graphs)

print(model.get_embedding())
