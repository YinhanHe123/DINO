from sknetwork.ranking import PageRank as pr
import numpy as np
import scipy.sparse as sp

from measures.base_measure import BaseMeasure

class PageRank(BaseMeasure):
    def __init__(self, *args) :
        super().__init__("PageRank")
        
    def get_immunized_nodes(self, A, k):
        pagerank = pr()
        scores = pagerank.fit_predict(sp.csr_matrix(A))
        selected_nodes = np.argsort(scores)[-k:]
        return selected_nodes.tolist()