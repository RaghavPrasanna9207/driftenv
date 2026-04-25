import matplotlib
matplotlib.use('TkAgg')

import networkx as nx
import matplotlib.pyplot as plt
from environment.graph_engine import GraphEngine

print("Script started")

engine = GraphEngine()
engine.load_graph("graph_templates/base_graph.json")
graph = engine.graph

print("Nodes:", graph.number_of_nodes())
print("Edges:", graph.number_of_edges())

plt.figure(figsize=(10, 7))
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=800, arrows=True)

print("Saving graph...")
plt.savefig("graph.png")

print("Done")
