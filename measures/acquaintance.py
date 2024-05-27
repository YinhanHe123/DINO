import networkx as nx
import random

from measures.base_measure import BaseMeasure

class Acquaintance(BaseMeasure):
    def __init__(self,*args) :
        super().__init__("Acquaintance")
        
    def get_immunized_nodes(self, A, k):
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        selected_nodes = []
        random.seed(33)
        while len(selected_nodes) < k:
            # Randomly select a node from the graph
            node = random.choice(list(G.nodes))

            # Get neighbors of the selected node
            neighbors = list(G.neighbors(node))

            # If there are no neighbors, continue with the next iteration
            if not neighbors:
                continue

            # Randomly select a neighbor
            neighbor = random.choice(neighbors)

            # Add the selected neighbor to the list of selected nodes, if it's not already there
            if neighbor not in selected_nodes:
                selected_nodes.append(neighbor)
        return selected_nodes