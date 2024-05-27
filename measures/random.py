import numpy as np

from measures.base_measure import BaseMeasure

class Random(BaseMeasure):
    def __init__(self, *args) :
        super().__init__("Random")
        
    def get_immunized_nodes(self, A, k):
        np.random.seed(30)
        num_nodes = A.shape[0]
        selected_nodes = np.random.choice(num_nodes, k, replace=False)
        return selected_nodes.tolist()
        