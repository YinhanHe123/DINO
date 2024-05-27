import os
import networkx as nx
import pandas as pd
import numpy as np
import scipy.sparse as sp

ROOT_PATH = os.path.dirname(os.path.abspath(__file__)).split("DINO")[0] + "DINO/"

def get_data(dataset, node_number = 1000, erdos_renyi_density=0.01):
    if dataset == 'hiv_transmission':
        file_path = ROOT_PATH + "data/hiv_transmission.tsv"
        data = pd.read_csv(file_path, sep='\t', usecols=['ID1', 'ID2'])
        G = nx.DiGraph()
        for _, row in data.iterrows():
            G.add_edge(row['ID1'], row['ID2'])
        A = sp.csr_array(nx.to_scipy_sparse_array(G)).asfptype()
    elif dataset == 'erdos_renyi':
        G = nx.erdos_renyi_graph(node_number,p=erdos_renyi_density, seed=200, directed=True)
        A = sp.csr_array(nx.to_scipy_sparse_array(G)).asfptype()
    else:
        with open(f"{ROOT_PATH}data/{dataset}.txt", 'r') as f:
            lines = f.readlines()
        row = []
        col = []
        for line in lines:
            if line[0] == '#':continue
            start, end = line.split('\t')
            row.append(int(start))
            col.append(int(end))
        max_nodes = max(max(set(row)), max(set(col))) + 1
        A = sp.csr_array((np.ones(len(row)), (row, col)), shape=(max_nodes, max_nodes)).asfptype()
    return A