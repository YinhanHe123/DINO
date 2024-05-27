import time
from matplotlib import pyplot as plt, rcParams
import networkx as nx
import EoN
import numpy as np
from collections import defaultdict
from measures.PageRank import pagerank_alg
from measures.DegreeDirect import degree_direct_alg
from measures.Random import random_alg
from measures.Hits import hits_alg
from measures.SCC import scc_alg
from measures.Closeness import closeness_centrality_alg
from measures.KatzCentrality import katz_centrality_alg
from measures.Acquaintance import acquaintance_alg
from measures.CBF import cbf_alg
import csv
import random
import scipy.sparse as sp

def get_data(dataset):
    if dataset == 'bitcoin_union':
        union_graph = nx.DiGraph()
        for bitcoin_dataset in ['bitcoin_alpha', 'bitcoin_otc']:
            with open("./data/" + bitcoin_dataset + ".csv", 'r') as f:
                node_num = {'bitcoin_alpha': 7604, 'bitcoin_otc': 6005}
                csvreader = csv.reader(f)
                edge_list = []
                for row in csvreader:
                    edge = (int(row[0]) - 1, int(row[1]) - 1, abs(int(row[2])))
                    edge_list.append(edge)
            BitcoinGraph = nx.DiGraph()
            BitcoinGraph.add_nodes_from(range(node_num[bitcoin_dataset]))
            BitcoinGraph.add_weighted_edges_from(edge_list)
            union_graph = nx.disjoint_union(union_graph, BitcoinGraph)
        A = nx.adjacency_matrix(union_graph).asfptype()
    elif dataset == 'C-elegans-frontal':
        node_set = set()
        node_dict = {}
        A = np.zeros((30000, 30000))
        with open("./data/C-elegans-frontal.txt", 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line[0] == '#': continue
                edge = line.split(' ')
                head, tail = int(edge[0]), int(edge[1])

                if head not in node_set:
                    node_dict[head] = len(node_set)
                    node_set.add(head)
                if tail not in node_set:
                    node_dict[tail] = len(node_set)
                    node_set.add(tail)
                A[node_dict[head], node_dict[tail]] = 1

            A = A[:len(node_set), :len(node_set)]
    elif dataset in ['soc-pokec-relationships','Email-EuAll',
                    'p2p-Gnutella08','soc-sign-epinions','web-BerkStan', 'p2p-Gnutella04', 'p2p-Gnutella05', 'p2p-Gnutella06', 'p2p-Gnutella09']:
        row = []
        col = []
        node_num={'soc-pokec-relationships':1632804, 'Email-EuAll':265214,
                'p2p-Gnutella08':6301, 'soc-sign-epinions':131828,'web-BerkStan':685231,'p2p-Gnutella09':8130}
        with open("./data/"+dataset+".txt", 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line[0] == '#':continue
                edge = line.split('\t')
                if dataset != 'soc-sign-epinions':
                    head, tail = int(edge[0]), int(edge[1])
                else:
                    head, tail, _ = int(edge[0]), int(edge[1]), int(edge[2])
                row.append(head)
                col.append(tail)
            data = np.ones(len(row))
            #print(max(row))
            A = sp.csr_array((data, (row, col)), shape=(node_num[dataset],node_num[dataset]))

    elif dataset in ['Cit-HepTh', 'C-elegans-frontal', 'Wiki-Vote', 'soc-Epinions1', 'Slashdot0902']:
        nodes_per_data = {'Cit-HepTh': 27770, 'C-elegans-frontal': 131, 'Wiki-Vote': 7115, 'soc-Epinions1': 75879,
                        'Slashdot0902': 82168, 'soc-pokec-relationships': 1632803}
        nodes_per_data = {'Cit-HepTh': 27770, 'C-elegans-frontal': 131, 'Wiki-Vote': 7115, 'soc-Epinions1': 75879,
                      'Slashdot0902': 82168, 'soc-pokec-relationships': 1632803}
        node_set = set()
        node_dict = {}
        A = np.zeros((nodes_per_data[dataset], nodes_per_data[dataset]))
        with open("./data/" + dataset + ".txt", 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line[0] == '#': continue
                edge = line.split('\t')
                head, tail = int(edge[0]), int(edge[1])

                if head not in node_set:
                    node_dict[head] = len(node_set)
                    node_set.add(head)
                if tail not in node_set:
                    node_dict[tail] = len(node_set)
                    node_set.add(tail)

                A[node_dict[head], node_dict[tail]] = 1

            A = A[:len(node_set), :len(node_set)]
    G = nx.from_numpy_array(A, create_using = nx.DiGraph)
    return G, A

def execute_alg(method, G, A, k):
    if method == 'Centrality':
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        immunized_nodes_list, run_times = centrality_alg(G, k)

        Lam_A_immunized_list = []
        for immunized_nodes in immunized_nodes_list:
            G_=G.copy()
            G_.remove_nodes_from(immunized_nodes)
            A_subgraph = nx.adjacency_matrix(G_)
            try:
                Lam_A_immunized, _ = sp.linalg.eigs(A_subgraph, 1, which='LR')
                Lam_A_immunized = abs(Lam_A_immunized).real
                print("The immunized spectrum radius is ", Lam_A_immunized)
            except:
                Lam_A_immunized = 0
            Lam_A_immunized_list.append(Lam_A_immunized)
        # print("immunized nodes of ", args.dataset, "by method", method, "is", immunized_nodes)
        # print("spectrum radius drop of ", args.dataset, "by method", method, "is ", radius_drop)
        return Lam_A_immunized_list, run_times
    else:
        if method == 'Acquaintance':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = acquaintance_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by Acquaintance:', end_finding - start_finding)
        elif method == 'LT':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = lt_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by LT:', end_finding - start_finding)
        elif method == 'IC':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = ic_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by IC:', end_finding - start_finding)
        elif method == 'CBF':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = cbf_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by CBF:', end_finding - start_finding)

        if method == 'CONTAIN':
            start_finding = time.time()
            immunized_nodes = contain_alg(A, k, args.r)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by CONTAIN:', end_finding - start_finding)
        elif method == 'DegreeIterative':
            start_finding = time.time()
            immunized_nodes = degree_iterative_alg(A, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by DegreeIterative:', end_finding - start_finding)
        elif method == 'DegreeDirect':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = degree_direct_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by DegreeDirect:', end_finding - start_finding)
        elif method == 'PageRank':
            #G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = pagerank_alg(A, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by PageRank:', end_finding - start_finding)
        elif method == 'Shi':
            start_finding = time.time()
            immunized_nodes = shi_alg(A, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by Shi:', end_finding - start_finding)
        elif method == 'Random':
            start_finding = time.time()
            immunized_nodes = random_alg(A, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by Random:', end_finding - start_finding)
        elif method == 'Walk':
            start_finding = time.time()
            immunized_nodes = walk_alg(A, k, args.walk_length)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by Walk:', end_finding - start_finding)
        elif method == 'Hits':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = hits_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by Hits:', end_finding - start_finding)
        elif method == 'BruteForce':
            start_finding = time.time()
            immunized_nodes = brute_force_alg(A,k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by BruteForce:', end_finding - start_finding)
        elif method == 'NewWalk':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = new_walk_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by NewWalk:', end_finding - start_finding)
        elif method == 'd_n_direct':
            start_finding = time.time()
            immunized_nodes = d_n_direct_alg(A, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by NewWalk:', end_finding - start_finding)
        elif method == 'Cycle':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            """
            mapping = {}
            nondec_nodes = [k for k, v in sorted(dict(G.degree()).items(), key=lambda item: item[1], reverse=True)]
            for i in range(len(nondec_nodes)):
                mapping[i]=nondec_nodes[i]
            G = nx.relabel_nodes(G, mapping, copy=True)
            """
            start_finding = time.time()
            immunized_nodes = cycle_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            """
            new_immu_nodes = []
            for node in immunized_nodes:
                new_immu_nodes.append(np.where(immunized_nodes == node)[0])
            """
            print('time for finding immunization nodes by Cycle:', end_finding - start_finding)
        elif method == 'Cycle_Dmax':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = cycle_dmax_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by Cycle_Dmax:', end_finding - start_finding)
        elif method == 'SCC':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, args.centrality_mode, args.scc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by SCC:', end_finding - start_finding)
        elif method == 'SCC_all':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            immunized_nodes, run_times = [], []
            for cent_mode in ['Random', 'PageRank', 'DegreeDirect', 'Hits', 'ext_perception',
                              # 'eigenvector_centrality',
                            # 'closeness_centrality',
                            # 'betweenness_centrality'
                              ]:
                start_finding = time.time()
                nodes_selected_by_this_method = scc_alg(G, k, cent_mode, args.scc_measure)
                immunized_nodes.append(nodes_selected_by_this_method)
                print("Immunized nodes by SCC-", cent_mode, ":", immunized_nodes[-1])
                end_finding = time.time()
                run_times.append(end_finding - start_finding)
                print('time for finding immunization nodes by SCC-', cent_mode, ': ', end_finding - start_finding)
        elif method == 'KSCC':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = kscc_alg(G, k, args.kscc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by KSCC:', end_finding - start_finding)
        elif method == 'BetweennessCentrality':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = betweenness_centrality_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by BetweennessCentrality:', end_finding - start_finding)
        elif method == 'ClosenessCentrality':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = closeness_centrality_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by ClosenessCentrality:', end_finding - start_finding)
        elif method == 'KatzCentrality':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = katz_centrality_alg(G,k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by KatzCentrality:', end_finding - start_finding)
        elif method == 'EigenvectorCentrality':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = eigenvector_centrality_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by EigenvectorCentrality:', end_finding - start_finding)
        elif method == 'ExpectedForce':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = expected_force_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by ExpectedForce:', end_finding - start_finding)
        elif method == 'new_SCC':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = new_scc_alg(G, k, args.centrality_mode, args.batch, args.scc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by new_SCC:', end_finding - start_finding)
        elif method == 'ext_perception':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = ext_perception_alg(G, k)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by ext_perception:', end_finding - start_finding)
        elif method == 'SCC-EigenvectorCentrality':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, 'eigenvector_centrality', args.scc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by SCC-EigenvectorCentrality:', end_finding - start_finding)
        elif method == 'SCC-ClosenessCentrality':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, 'closeness_centrality', args.scc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by SCC-ClosenessCentrality:', end_finding - start_finding)
        elif method == 'SCC-BetweennessCentrality':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, 'betweenness_centrality', args.scc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by SCC-BetweennessCentrality:', end_finding - start_finding)
        elif method == 'SCC-KatzCentrality':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, 'katz_centrality', args.scc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by SCC-KatzCentrality:', end_finding - start_finding)
        elif method == 'SCC-Random':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, 'Random', args.scc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by SCC-Random:', end_finding - start_finding)
        elif method == 'SCC-PageRank':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, 'PageRank', args.scc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by SCC-PageRank', end_finding - start_finding)
        elif method == 'SCC-DegreeDirect':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, 'DegreeDirect', args.scc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by SCC-DegreeDirect', end_finding - start_finding)
        elif method == 'SCC-Hits':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, 'Hits', args.scc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by SCC-Hits', end_finding - start_finding)
        elif method == 'SCC-ext_perception':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, 'ext_perception', "spectrum_radius")
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by SCC-ExPSCC', end_finding - start_finding)
        elif method == 'DINO':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, 'dino', "spectrum_radius")
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by DINO', end_finding - start_finding)

        if method != 'SCC_all':
            print("Immunized nodes by", method, ":", immunized_nodes)
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            G.remove_nodes_from(immunized_nodes)

            A_subgraph = nx.adjacency_matrix(G)
            try:
                Lam_A_immunized, _ = sp.linalg.eigs(A_subgraph, 1, which='LR')
                Lam_A_immunized = abs(Lam_A_immunized).real
                print("The immunized spectrum radius is ", Lam_A_immunized)
            except:
                Lam_A_immunized = 0
    return immunized_nodes

def run_epi(immunized_G, epi_model, init_infected_rate, recovery_rate, e_to_i_rate, trans_rate):
    if epi_model == 'SIS':
        t_imm, _, I_imm = EoN.basic_discrete_SIS(immunized_G, p=trans_rate, rho=init_infected_rate,tmax=100)
        t_imm, I_imm = t_imm[1:], I_imm[1:]
    elif epi_model == 'SIR':
        H = nx.DiGraph()
        H.add_node('S')
        H.add_edge('I', 'R', rate=recovery_rate)
        J = nx.DiGraph()
        J.add_edge(('I', 'S'), ('I', 'I'), rate=trans_rate)
        IC = defaultdict(lambda: 'S')
        sample_number = int(immunized_G.number_of_nodes() * init_infected_rate)
        initial_infected_nodes = random.sample(list(immunized_G.nodes()), sample_number)
        for node in initial_infected_nodes:
            IC[node] = 'I'
        return_statuses = ('S', 'I', 'R')
        t_imm, _, I_imm, _ = EoN.Gillespie_simple_contagion(immunized_G, H, J, IC, return_statuses,tmax=100)
        t_imm, I_imm = t_imm[-20:], I_imm[-20:]
    elif epi_model == 'SEIR':
        H = nx.DiGraph()
        H.add_node('S')
        H.add_edge('E', 'I', rate=e_to_i_rate)
        H.add_edge('I', 'R', rate=recovery_rate)
        J = nx.DiGraph()
        J.add_edge(('I', 'S'), ('I', 'E'), rate=trans_rate)
        IC = defaultdict(lambda: 'S')
        sample_number = int(immunized_G.number_of_nodes() * init_infected_rate)
        initial_infected_nodes = random.sample(list(immunized_G.nodes()), sample_number)
        for node in initial_infected_nodes:
            IC[node] = 'I'
        return_statuses = ('S', 'E', 'I', 'R')
        t_imm, _, _, I_imm, _ = EoN.Gillespie_simple_contagion(immunized_G, H, J, IC, return_statuses,tmax=100)
        t_imm, I_imm = t_imm[-20:], I_imm[-20:]
    return t_imm, I_imm

def run_epi_org(immunized_G, epi_model, init_infected_rate, recovery_rate, e_to_i_rate, trans_rate):
    if epi_model == 'SIS':
        t_imm, _, I_imm = EoN.basic_discrete_SIS(immunized_G, p=trans_rate, rho=init_infected_rate,tmax=100)
    elif epi_model == 'SIR':
        H = nx.DiGraph()
        H.add_node('S')
        H.add_edge('I', 'R', rate=recovery_rate)
        J = nx.DiGraph()
        J.add_edge(('I', 'S'), ('I', 'I'), rate=trans_rate)
        IC = defaultdict(lambda: 'S')
        sample_number = int(immunized_G.number_of_nodes() * init_infected_rate)
        initial_infected_nodes = random.sample(list(immunized_G.nodes()), sample_number)
        for node in initial_infected_nodes:
            IC[node] = 'I'
        return_statuses = ('S', 'I', 'R')
        t_imm, _, I_imm, _ = EoN.Gillespie_simple_contagion(immunized_G, H, J, IC, return_statuses,tmax=100)
    elif epi_model == 'SEIR':
        H = nx.DiGraph()
        H.add_node('S')
        H.add_edge('E', 'I', rate=e_to_i_rate)
        H.add_edge('I', 'R', rate=recovery_rate)
        J = nx.DiGraph()
        J.add_edge(('I', 'S'), ('I', 'E'), rate=trans_rate)
        IC = defaultdict(lambda: 'S')
        sample_number = int(immunized_G.number_of_nodes() * init_infected_rate)
        initial_infected_nodes = random.sample(list(immunized_G.nodes()), sample_number)
        for node in initial_infected_nodes:
            IC[node] = 'I'
        return_statuses = ('S', 'E', 'I', 'R')
        t_imm, _, _, I_imm, _ = EoN.Gillespie_simple_contagion(immunized_G, H, J, IC, return_statuses,tmax=100)
    return t_imm, I_imm

def run_simulation(dataset, epi_model, k=20, init_infected_rate=0.95, recovery_rate=0.8, e_to_i_rate=0.9, trans_rate=0.03):
    G,A = get_data(dataset)
    t_org, i_org = run_epi_org(G, epi_model, init_infected_rate, recovery_rate, e_to_i_rate, trans_rate)
    methods = ['Random', 'DegreeDirect', 'KatzCentrality', 'ClosenessCentrality', 'Hits', 'PageRank', 'Acquaintance', 'CBF', 'DINO']
    labels = ['Random', 'Degree', 'Katz.', 'Close.', 'HITS', 'PageRank', 'Acqu.', 'CBF', 'DINO']
    colors = ['#D57DBF', '#7C534A', '#8D69B8', '#C53A32', '#519E3E', '#3B75AF', '#EF8636', '#28cc51', '#28aecc']
    markers = ['^', 'D', 's', 'P', 'o', 'v', '<', 'x', '>']
    output = []
    for idx in range(len(methods)):
        immunized_nodes = execute_alg(methods[idx], G.copy(), A, k)
        immu_G = G.copy()
        immu_G.remove_nodes_from(immunized_nodes)
        t, i = run_epi(immu_G, epi_model, init_infected_rate, recovery_rate, e_to_i_rate, trans_rate)
        final_t = np.concatenate((t[::3], np.array([t[-1]])))
        final_i = np.concatenate((i[::3], np.array([i[-1]])))
        output.append({"method": methods[idx], "t": final_t.tolist(), "i":final_i.tolist()})
        ax.plot(final_t, final_i, label=labels[idx], marker=markers[idx], markersize=10, color=colors[idx])
    print("orig")
    print(t_org.tolist(), i_org.tolist())
    return output
    
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 25

model = "SIR"

fig, ax = plt.subplots(figsize=(9, 8))
res = run_simulation("Wiki-Vote", model)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color="grey")
plt.gca().set_facecolor("#eeeeee")
plt.tight_layout()
plt.legend()
ax.set_xlabel("time", fontname='Times New Roman', fontsize=28)
ax.set_ylabel("Infected people", fontname='Times New Roman', fontsize=35)
plt.savefig(f'model{model}.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(res)
