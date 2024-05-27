from networkx import betweenness_centrality
import networkx as nx

from measures.base_measure import BaseMeasure
from measures.measure_utils import get_largest_dict_keys

class BetweennessCentrality(BaseMeasure):
    def __init__(self, *args) :
        super().__init__("BetweennessCentrality")
        
    def get_immunized_nodes(self, A, k):
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        score_dict = dict(betweenness_centrality(G))
        print("a")
        selected_nodes = get_largest_dict_keys(score_dict, k)
        return selected_nodes