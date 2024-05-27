from networkx import katz_centrality
import networkx as nx

from measures.base_measure import BaseMeasure
from measures.measure_utils import get_largest_dict_keys

class KatzCentrality(BaseMeasure):
    def __init__(self, *args) :
        super().__init__("KatzCentrality")
        
    def get_immunized_nodes(self, A, k):
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        score_dict = katz_centrality(G, alpha=0.001,max_iter=5000)
        selected_nodes = get_largest_dict_keys(score_dict, k)
        return selected_nodes
        