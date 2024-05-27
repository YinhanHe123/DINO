import networkx as nx

from measures.base_measure import BaseMeasure
from measures.measure_utils import get_largest_dict_keys

class DegreeDirect(BaseMeasure):
    def __init__(self, *args) :
        super().__init__("DegreeDirect")
        
    def get_immunized_nodes(self, A, k):
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        degree_dict = dict(G.degree())
        selected_nodes = get_largest_dict_keys(degree_dict, k)
        return selected_nodes