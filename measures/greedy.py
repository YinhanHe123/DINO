import networkx as nx
import numpy as np
import scipy.sparse as sp

from measures.base_measure import BaseMeasure
from measures.measure_utils import get_largest_dict_keys

class Greedy(BaseMeasure):
    def __init__(self, *args) :
        super().__init__("Greedy")
    
    def get_node_to_immunize(self, A, nodes_available):
        radii_list = np.zeros(A.shape[0]) + 1e4
        for node in nodes_available:
            A_ = A.copy()
            A_[node,:]=0
            A_[:,node]=0
            try:
                score, _ = sp.linalg.eigs(A_, k=1, which = 'LR')
                score = abs(score.item())
            except:
                score = 0
            radii_list[node] = score
        node_to_delete = np.argmin(radii_list)
        A[node_to_delete,:]=0
        A[:, node_to_delete]=0
        return node_to_delete, A
        
    def get_immunized_nodes(self, A, k):
        num_nodes = A.shape[0]
        nodes_available = list(range(num_nodes))
        selected_nodes = []
        A_ = A.copy()
        for _ in range(k):
            node_to_delete, A_ = self.get_node_to_immunize(A_, nodes_available)
            nodes_available.remove(node_to_delete)
            selected_nodes.append(node_to_delete)
        return selected_nodes
