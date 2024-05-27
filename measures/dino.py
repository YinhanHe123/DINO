import networkx as nx
import numpy as np
import scipy.sparse as sp

from measures.base_measure import BaseMeasure
from measures.measure_utils import get_largest_dict_keys

class Dino(BaseMeasure):
    def __init__(self, *args) :
        super().__init__("DINO")
        
    def build_scc_and_measure_lists(self,G):
        # Find all SCCs with >= 2 nodes and calculate their spectral radii
        SCCs = list(nx.strongly_connected_components(G))
        SCC_list = []
        measure_number_list = []
        for SCC in SCCs:
            if len(SCC) <= 1:
                continue
            SCC_induced_subgraph = nx.subgraph(G, SCC)
            SCC_induced_subgraph_adj = nx.to_numpy_array(SCC_induced_subgraph)
            SCC_list.append(SCC)
            try:
                measure_number_list.append(abs(sp.linalg.eigs(SCC_induced_subgraph_adj,
                                                                1, which='LR')[0].item()))
            except:
                measure_number_list.append(1)
        # Sort the dictionary by non-increasing order of measure numbers.
        sorted_SCC_list_index = np.argsort(-np.array(measure_number_list))
        SCC_list = [SCC_list[i] for i in sorted_SCC_list_index]
        measure_number_list = [measure_number_list[i] for i in sorted_SCC_list_index]
        return SCC_list, measure_number_list
    
    def merge_sorted_scc_and_measure_lists(self, scc_1, measure_1, scc_2, measure_2):
        merged_scc, merged_measure = [], []
        i,j=0,0
        while i < len(measure_1) and j < len(measure_2):
            if measure_1[i] > measure_2[j]:
                merged_scc.append(scc_1[i])
                merged_measure.append(measure_1[i])
                i += 1
            else:
                merged_scc.append(scc_2[j])
                merged_measure.append(measure_2[j])
                j += 1
        merged_measure = merged_measure + measure_1[i:] + measure_2[j:]
        merged_scc = merged_scc + scc_1[i:] + scc_2[j:]
        return merged_scc, merged_measure    

    def calculate_dino_score(self, G):
        # first calculate the degree of each node
        in_degree, out_degree = dict(G.in_degree()), dict(G.out_degree())
        inner_prod = np.dot(list(in_degree.values()), list(out_degree.values()))
        S = sum(list(in_degree.values()))
        # calculate the app spectral radius when every node is removed
        app_spectral_radius = {}
        for node in G.nodes:
            # calculate the revised degree sequence after removing the node
            # it means that the in_degree and out_degree of the node is removed
            # besides, the in_degree of the nodes that have an edge to the node is decreased by 1
            # the out_degree of the nodes that have an edge from the node is decreased by 1
            # then plus the number of nodes which are both its predecessors and successors
            inner_prod_v, S_v = inner_prod, S
            inner_prod_v = inner_prod_v - (in_degree[node]*out_degree[node]+sum([in_degree[pre] for pre in G.predecessors(node)])+
                        sum([out_degree[suc] for suc in G.successors(node)])) + len(set(G.predecessors(node)).intersection(G.successors(node)))
            S_v -= (in_degree[node]+out_degree[node])
            app_spectral_radius[node] = -inner_prod_v/S_v
        return app_spectral_radius
    
        
    def get_immunized_nodes(self, A, k):
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        selected_nodes = []
        SCC_list, measure_number_list = self.build_scc_and_measure_lists(G)
        
        for _ in range(k):
            if not bool(SCC_list): # no SCCs
                return selected_nodes
            # Pick the largest SCC and remove a node
            SCC_induced_subgraph = nx.DiGraph(nx.subgraph(G, SCC_list[0]))
            score_dict = self.calculate_dino_score(SCC_induced_subgraph)    
            node_to_delete = get_largest_dict_keys(score_dict, 1)[0]

            SCC_induced_subgraph.remove_node(node_to_delete)
            selected_nodes.append(node_to_delete)

            # Remove selected SCC
            SCC_list = SCC_list[1:]
            measure_number_list = measure_number_list[1:]

            # Calculate the SCCs in SCC_induced_subgraph
            sub_SCCs, measure_number_of_sub_SCCs = self.build_scc_and_measure_lists(SCC_induced_subgraph)
            SCC_list, measure_number_list = self.merge_sorted_scc_and_measure_lists(SCC_list, measure_number_list,
                                                                              sub_SCCs, measure_number_of_sub_SCCs)
        return selected_nodes