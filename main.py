from collections import defaultdict
import json
import os
import random
import networkx as nx
import scipy.sparse as sp
import time
import EoN

from utils import get_args
from data.datatset import ROOT_PATH, get_data
from measures.measure_utils import MEASURE_MAP, get_measure

def get_immunized_radius(A, immunized_nodes):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    G.remove_nodes_from(immunized_nodes)
    A_immunized = sp.csr_array(nx.to_scipy_sparse_array(G)).asfptype()
    try:
        Lam_A_immunized, _ = sp.linalg.eigs(A_immunized, 1, which='LR')
        Lam_A_immunized = abs(Lam_A_immunized).real[0]
    except:
        Lam_A_immunized = 0
    return Lam_A_immunized

def save_results(dataset, k, immunized_nodes, orig_radius, new_radius, tot_time, measure):
    file_path = f"{ROOT_PATH}saved_results/{dataset}_{measure}.csv"
    if not os.path.isfile(file_path):
        with open(file_path, "a+") as f:
            f.write("k, original spectral radius, immunized spectral radius, radius drop, immunized nodes, total time\n")
    with open(file_path, "a+") as f:
        f.write(f"{k},{orig_radius},{new_radius},{orig_radius - new_radius}, {immunized_nodes},{tot_time}\n")
        
def simulate_empidemic(epidemic_type, G, init_infected_rate, recovery_rate, e_to_i_rate, trans_rate, orig=False):
    if epidemic_type == 'SIS':
        t, _, i = EoN.basic_discrete_SIS(G, p=trans_rate, rho=init_infected_rate,tmax=60)
        if not orig:
            t, i = t[1:], i[1:]
    elif epidemic_type == 'SIR':
        H = nx.DiGraph()
        H.add_node('S')
        H.add_edge('I', 'R', rate=recovery_rate)
        J = nx.DiGraph()
        J.add_edge(('I', 'S'), ('I', 'I'), rate=trans_rate)
        IC = defaultdict(lambda: 'S')
        sample_number = int(G.number_of_nodes() * init_infected_rate)
        random.seed(33)
        initial_infected_nodes = random.sample(list(G.nodes()), sample_number)
        for node in initial_infected_nodes:
            IC[node] = 'I'
        return_statuses = ('S', 'I', 'R')
        t, _, i, _ = EoN.Gillespie_simple_contagion(G, H, J, IC, return_statuses, tmax=100)
        if not orig:
            t, i = t[-20:], i[-20:]
    elif epidemic_type == 'SEIR':
        H = nx.DiGraph()
        H.add_node('S')
        H.add_edge('E', 'I', rate=e_to_i_rate)
        H.add_edge('I', 'R', rate=recovery_rate)
        J = nx.DiGraph()
        J.add_edge(('I', 'S'), ('I', 'E'), rate=trans_rate)
        IC = defaultdict(lambda: 'S')
        sample_number = int(G.number_of_nodes() * init_infected_rate)
        initial_infected_nodes = random.sample(list(G.nodes()), sample_number)
        for node in initial_infected_nodes:
            IC[node] = 'I'
        return_statuses = ('S', 'E', 'I', 'R')
        t, _, _, i, _ = EoN.Gillespie_simple_contagion(G, H, J, IC, return_statuses, tmax=100)
        if not orig:
            t, i = t[-20:], i[-20:]
    return t, i

def run_epidemic_simulation(epidemic_type, A, args, k, init_infected_rate=0.95, recovery_rate=0.8, e_to_i_rate=0.9, trans_rate=0.03):
    G = nx.from_numpy_array(A, create_using = nx.DiGraph)
    orig_t, orig_i = simulate_empidemic(epidemic_type, G, init_infected_rate, recovery_rate, e_to_i_rate, trans_rate, orig=True)
    res = {"original": {"t": orig_t.tolist(), "i": orig_i.tolist()}}
    for m in ['random', 'dino', 'closeness_centrality', 'hits', 'degree_direct', 'page_rank', 'cbf', 'katz_centrality', 'acquaintance']:
        measure = get_measure(m, args)
        immunized_nodes = measure.get_immunized_nodes(A, k)
        immunized_G = G.copy()
        immunized_G.remove_nodes_from(immunized_nodes)
        new_t, new_i = simulate_empidemic(epidemic_type, immunized_G, init_infected_rate, recovery_rate, e_to_i_rate, trans_rate)
        res[m] = {"t":new_t.tolist(), "i": new_i.tolist()}
    json.dump(res, open(f"{ROOT_PATH}saved_results/{epidemic_type}_{args.dataset}.json", "w"), indent=4)
    
def main():
    args = get_args()
    A = get_data(args.dataset)
    k = int(args.k_percent*A.shape[0]) if args.k is None else args.k
    
    if args.epidemic_type is not None:
        run_epidemic_simulation(args.epidemic_type, A, args, k)
    else:
        Lam_A, _ = sp.linalg.eigs(A, 1, which='LR')
        Lam_A = abs(Lam_A).real[0]
        print("Spectral radius of the original graph is", Lam_A)
        
        measure = get_measure(args.measure, args)
        start_time = time.time()
        immunized_nodes = measure.get_immunized_nodes(A, k)
        Lam_A_immunized = get_immunized_radius(A, immunized_nodes)
        radius_drop = Lam_A - Lam_A_immunized
        tot_time = time.time() - start_time
        print(f"Spectral radius drop of {args.dataset} by measure {args.measure} is {radius_drop} in {tot_time} seconds")
        save_results(args.dataset, k, immunized_nodes, Lam_A, Lam_A_immunized, tot_time, measure.measure_name)
    
if __name__== "__main__":
    main()