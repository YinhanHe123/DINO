import networkx as nx

from measures.base_measure import BaseMeasure
from measures.measure_utils import get_largest_dict_keys

class Hits(BaseMeasure):
    def __init__(self, *args) :
        super().__init__("HITS")
        
    def get_immunized_nodes(self, A, k):
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        selected_nodes = []
        for _ in range(k):
            score_dict, _ = nx.hits(G)
            selected_node = get_largest_dict_keys(score_dict, 1)
            selected_nodes.append(selected_node[0])
            G.remove_node(selected_node[0])
        return selected_nodes