import networkx as nx
import EoN
import matplotlib.pyplot as plt
import argparse
from matplotlib import rcParams
import numpy as np
from collections import defaultdict
from measures.CONTAIN import contain_alg
from measures.DegreeIterative import degree_iterative_alg
import pandas as pd
from measures.PageRank import pagerank_alg
from measures.DegreeDirect import degree_direct_alg
from measures.Shi import shi_alg
from measures.Random import random_alg
from measures.Walk import walk_alg
from measures.BruteForce import brute_force_alg
from measures.NewWalk import new_walk_alg
from measures.Ext_perception import ext_perception_alg
from measures.Hits import hits_alg
from measures.d_n_direct import d_n_direct_alg
from measures.Cycle import cycle_alg
from measures.Cycle_Dmax import cycle_dmax_alg
from measures.SCC import scc_alg, new_scc_alg
from measures.KSCC import kscc_alg
from measures.BetweennessCentrality import betweenness_centrality_alg
from measures.Closeness import closeness_centrality_alg
from measures.EigenvectorCentrality import eigenvector_centrality_alg
from measures.KatzCentrality import katz_centrality_alg
from measures.ExpectedForce import expected_force_alg
from measures.Centrality import centrality_alg
from measures.Acquaintance import acquaintance_alg
from measures.LT import lt_alg
from measures.IC import ic_alg
from measures.CBF import cbf_alg
import csv
import time
import random
import scipy.sparse as sp
# this time we simulate with every node immunization method.
parser = argparse.ArgumentParser(description='simulate the epidemiological development')
parser.add_argument('--num_nodes', type= int, default = 100, help='number of nodes in the epidemical network')
parser.add_argument('--method', type=str, default='all_methods', help='node immunization algorithms')
parser.add_argument('--k', type=int, default=None, help='the number of nodes for immunization')
parser.add_argument('--node_selection_rate', type=float, default=None, help='the proportion of node selection')
parser.add_argument('--r', type=int, default=60,
                    help='the number of eigenvalues considered for recovering the adjacency matrix')
parser.add_argument('--walk_length', type=int, default=5,
                    help='The largest length of walk considered in Walk algorithm')
parser.add_argument('--dataset', type=str,default='erdos_renyi')
parser.add_argument('--init_infecteds',type=float, default=0.95,
                    help='the rate of people being infected initially' )
parser.add_argument('--trans_rate', type=float, default=0.03, #previous:0.03
                    help='transmission rate')
parser.add_argument('--recover_rate',type=float,default=0.2,
                    help='recover rate')
parser.add_argument('--model', type=str, default='SIS',
                    help='epidemic propagation model')

parser.add_argument('--entrance_to_infected_rate', type=float, default=0.9,
                    help='the probability that E turns to I')
parser.add_argument('--scc_measure', type=str, default='spectrum_radius',
                        help='choose how to rank SCCs as approximation of spectrum radius in SCC alg.')
args = parser.parse_args()

#t_dict = {}
#I_dict = {}
# generate graph: we do erdos-renyi graphs
if args.dataset == 'bitcoin_union':
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
elif args.dataset == 'erdos_renyi':
    G = nx.erdos_renyi_graph(args.num_nodes,0.3,directed=True)
    A = nx.adjacency_matrix(G).asfptype().todense()

elif args.dataset == 'C-elegans-frontal':
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
elif args.dataset in ['soc-pokec-relationships','Email-EuAll',
                   'p2p-Gnutella08','soc-sign-epinions','web-BerkStan', 'p2p-Gnutella04', 'p2p-Gnutella05', 'p2p-Gnutella06', 'p2p-Gnutella09']:
    row = []
    col = []
    node_num={'soc-pokec-relationships':1632804, 'Email-EuAll':265214,
              'p2p-Gnutella08':6301, 'soc-sign-epinions':131828,'web-BerkStan':685231,'p2p-Gnutella09':8130}
    with open("./data/"+args.dataset+".txt", 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line[0] == '#':continue
            edge = line.split('\t')
            if args.dataset != 'soc-sign-epinions':
                head, tail = int(edge[0]), int(edge[1])
            else:
                head, tail, _ = int(edge[0]), int(edge[1]), int(edge[2])
            row.append(head)
            col.append(tail)
        data = np.ones(len(row))
        #print(max(row))
        A = sp.csr_array((data, (row, col)), shape=(node_num[args.dataset],node_num[args.dataset]))

elif args.dataset in ['Cit-HepTh', 'C-elegans-frontal', 'Wiki-Vote', 'soc-Epinions1', 'Slashdot0902']:
    nodes_per_data = {'Cit-HepTh': 27770, 'C-elegans-frontal': 131, 'Wiki-Vote': 7115, 'soc-Epinions1': 75879,
                      'Slashdot0902': 82168, 'soc-pokec-relationships': 1632803}
    node_set = set()
    node_dict = {}
    A = np.zeros((nodes_per_data[args.dataset], nodes_per_data[args.dataset]))
    with open("./data/" + args.dataset + ".txt", 'r') as f:
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

# original graph epidemical development.
#t, S, I = EoN.basic_discrete_SIS(G, 0.05,rho=0.85, tmax = 20)
#t,S,I = EoN.basic_discrete_SIS(G, p=args.trans_rate,rho=args.init_infecteds, tmax = 20)
if args.model == 'SIS':
    t, S, I = EoN.basic_discrete_SIS(G, p=args.trans_rate, rho=args.init_infecteds,
                                                 tmax=60)
elif args.model == 'SIR':
    H = nx.DiGraph()
    H.add_node('S')
    #H.add_edge('E', 'I', rate=args.entrance_to_infected_rate)
    H.add_edge('I', 'R', rate=args.recover_rate)

    J = nx.DiGraph()
    J.add_edge(('I', 'S'), ('I', 'I'), rate=args.trans_rate)
    IC = defaultdict(lambda: 'S')
    sample_number = int(G.number_of_nodes() * args.init_infecteds)
    random.seed(33)
    initial_infected_nodes = random.sample(list(G.nodes()), sample_number)
    for node in initial_infected_nodes:
        IC[node] = 'I'

    return_statuses = ('S', 'I', 'R')

    t, S, I, _ = EoN.Gillespie_simple_contagion(G, H, J, IC, return_statuses,
                                                   tmax=100)
elif args.model == 'SEIR':
    H = nx.DiGraph()
    H.add_node('S')
    H.add_edge('E', 'I', rate=args.entrance_to_infected_rate)
    H.add_edge('I', 'R', rate=args.recover_rate)

    J = nx.DiGraph()
    J.add_edge(('I', 'S'), ('I', 'E'), rate=args.trans_rate)
    IC = defaultdict(lambda: 'S')
    sample_number = int(G.number_of_nodes() * args.init_infecteds)
    initial_infected_nodes = random.sample(list(G.nodes()), sample_number)
    for node in initial_infected_nodes:
        IC[node] = 'I'

    return_statuses = ('S', 'E', 'I', 'R')

    t, S, E, I, _ = EoN.Gillespie_simple_contagion(G, H, J, IC, return_statuses,
                                                                   tmax=100)
plt.plot(t,I,label="Original network")

# def execute_alg(method):
#     if args.k != None:
#         k = args.k
#     else:
#         k = int(args.node_selection_rate*A.shape[0])
#     #if method == 'CONTAIN':
#     #    start_finding = time.time()
#     #    immunized_nodes = contain_alg(A, args.k, args.r)
#     #    end_finding = time.time()
#     #    print('time for finding immunization nodes by CONTAIN:', end_finding - start_finding)
#     #elif method == 'DegreeIterative':
#     #    start_finding = time.time()
#     #    immunized_nodes = degree_iterative_alg(A, args.k)
#     #    end_finding = time.time()
#     #    print('time for finding immunization nodes by DegreeIterative:', end_finding - start_finding)
#     if method == 'Centrality':
#         G = nx.from_numpy_array(A, create_using=nx.DiGraph)
#         immunized_nodes = centrality_alg(G, k)
#     if method == 'DegreeDirect':
#         start_finding = time.time()
#         G = nx.from_numpy_array(A, create_using=nx.DiGraph)
#         immunized_nodes = degree_direct_alg(G, k)
#         end_finding = time.time()
#         print('time for finding immunization nodes by DegreeDirect:', end_finding - start_finding)
#
#     elif method == 'Shi':
#         start_finding = time.time()
#         immunized_nodes = shi_alg(A, k)
#         end_finding = time.time()
#         print('time for finding immunization nodes by Shi:', end_finding - start_finding)
#
#     #elif method == 'NewWalk':
#     #    start_finding = time.time()
#     #    immunized_nodes = new_walk_alg(G, args.k)
#     #    end_finding = time.time()
#     #    print('time for finding immunization nodes by Walk:', end_finding - start_finding)
#     elif method == 'Random':
#         start_finding = time.time()
#         immunized_nodes = random_alg(A, k)
#         end_finding = time.time()
#         print('time for finding immunization nodes by Random:', end_finding - start_finding)
#
#     elif method == 'PageRank':
#         start_finding = time.time()
#         immunized_nodes = pagerank_alg(A, k)
#         end_finding = time.time()
#         print('time for finding immunization nodes by PageRank:', end_finding - start_finding)
#
#     elif method == 'Hits':
#         G = nx.from_numpy_array(A, create_using=nx.DiGraph)
#         start_finding = time.time()
#         immunized_nodes = hits_alg(G, k)
#         end_finding = time.time()
#         print('time for finding immunization nodes by Hits:', end_finding - start_finding)
#
#     elif method == 'SCC':
#         G = nx.from_numpy_array(A, create_using=nx.DiGraph)
#         start_finding = time.time()
#         immunized_nodes = scc_alg(G, k, centrality_mode='degree', scc_measure='spectrum_radius')
#         end_finding = time.time()
#         print('time for finding immunization nodes by SCC:', end_finding - start_finding)
#     return immunized_nodes

# do graph node immunization

def execute_alg(method, A, args):
    if args.k != None:
        k = args.k
    else:
        k = int(args.node_selection_rate*A.shape[0])
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
            immunized_nodes = scc_alg(G, k, 'ext_perception', args.scc_measure)
            end_finding = time.time()
            run_times = end_finding - start_finding
            print('time for finding immunization nodes by SCC-ExPSCC', end_finding - start_finding)
        elif method == 'DINO':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            start_finding = time.time()
            immunized_nodes = scc_alg(G, k, 'dino', args.scc_measure)
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
        # else:
        #     Lam_A_immunized = []
        #     cent_mode_list = ['Random', 'PageRank', 'DegreeDirect', 'Hits', 'ext_perception', 'eigenvector_centrality',
        #                     # 'closeness_centrality',
        #                     # 'betweenness_centrality'
        #                       ]
        #     for i, nodes in enumerate(immunized_nodes):
        #         G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        #         G.remove_nodes_from(nodes)
        #         A_subgraph = nx.adjacency_matrix(G)
        #         try:
        #             immu_lam, _ = sp.linalg.eigs(A_subgraph, 1, which='LR')
        #             immu_lam = abs(immu_lam).real
        #             print("The immunized spectrum radius of SCC-",cent_mode_list[i],"is ", immu_lam)
        #         except:
        #             immu_lam = 0
        #         Lam_A_immunized.append(immu_lam)

        return immunized_nodes
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 25

fig, ax = plt.subplots(figsize=(9, 8))
if args.method != 'all_methods':
    immu_G = G.copy()
    immunized_nodes = execute_alg(args.method)
    immu_G.remove_nodes_from(immunized_nodes)
    t_imm, S_imm, I_imm = EoN.basic_discrete_SIS(immu_G, p=args.trans_rate, rho=args.init_infecteds, tmax=100)
    plt.plot(t_imm, I_imm, label=args.method)
elif args.method == 'all_methods':
    output = []
    for method in ['Random', 'DegreeDirect', 'KatzCentrality', 'ClosenessCentrality', 'Hits', 'PageRank', 'Acquaintance', 'CBF', 'DINO']:
        if method == 'Centrality':
            immunized_node_list = execute_alg(method)
            for i in range(len(immunized_node_list)):
                centrality_name_list = [  # 'voterank',
                    # 'trophic_levels',
                    # 'degree_centrality', 'in_degree_centrality', 'out_degree_centrality',
                    'eigenvector_centrality', #'katz_centrality',
                    #'closeness_centrality',
                    #'betweenness_centrality',
                    # 'load_centrality',
                    # 'harmonic_centrality'
                ]
                immunized_nodes = immunized_node_list[i]
                immu_G = G.copy()
                immu_G.remove_nodes_from(immunized_nodes)
                # t_imm, S_imm, I_imm = EoN.basic_discrete_SIS(immu_G, 0.05, rho=0.85, tmax=20)
                #t_imm, S_imm, I_imm = EoN.basic_discrete_SIS(immu_G, 0.05, rho=0.95, tmax=20)
                if args.model == 'SIS':
                    t_imm, S_imm, I_imm = EoN.basic_discrete_SIS(immu_G, p=args.trans_rate, rho=args.init_infecteds,
                                                             tmax=100)
                elif args.model == 'SIR':
                    H = nx.DiGraph()
                    H.add_node('S')
                    #H.add_edge('E', 'I', rate=args.entrance_to_infected_rate)
                    H.add_edge('I', 'R', rate=args.recover_rate)

                    J = nx.DiGraph()
                    J.add_edge(('I', 'S'), ('I', 'I'), rate=args.trans_rate)
                    IC = defaultdict(lambda: 'S')
                    sample_number = int(immu_G.number_of_nodes() * args.init_infecteds)
                    initial_infected_nodes = random.sample(list(immu_G.nodes()), sample_number)
                    for node in initial_infected_nodes:
                        IC[node] = 'I'

                    return_statuses = ('S', 'I', 'R')

                    t_imm, S_imm, I_imm, _ = EoN.Gillespie_simple_contagion(immu_G, H, J, IC, return_statuses,
                                                                                   tmax=100)
                elif args.model == 'SEIR':
                    H = nx.DiGraph()
                    H.add_node('S')
                    H.add_edge('E', 'I', rate=args.entrance_to_infected_rate)
                    H.add_edge('I', 'R', rate=args.recover_rate)

                    J = nx.DiGraph()
                    J.add_edge(('I', 'S'), ('I', 'E'), rate=args.trans_rate)
                    IC = defaultdict(lambda: 'S')
                    sample_number = int(immu_G.number_of_nodes() * args.init_infecteds)
                    initial_infected_nodes = random.sample(list(immu_G.nodes()), sample_number)
                    for node in initial_infected_nodes:
                        IC[node] = 'I'

                    return_statuses = ('S', 'E', 'I', 'R')

                    t_imm, S_imm, E_imm, I_imm, _ = EoN.Gillespie_simple_contagion(immu_G, H, J, IC, return_statuses,
                                                                   tmax=100)

                #t_dict[centrality_name_list[i]] = t_imm
                #I_dict[centrality_name_list[i]] = I_imm
                if centrality_name_list[i] == 'eigenvector_centrality':
                    ax.plot(t_imm, I_imm, label='Eigenvec.Cen.')
                elif centrality_name_list[i] == 'katz_centrality':
                    ax.plot(t_imm, I_imm, label='Katz.Cen.')
                elif centrality_name_list[i] == 'closeness_centrality':
                    ax.plot(t_imm, I_imm, label='Close.Cen.')
                else:
                    ax.plot(t_imm, I_imm, label=centrality_name_list[i])
        else:
            immunized_nodes = execute_alg(method, A, args)
            immu_G = G.copy()
            immu_G.remove_nodes_from(immunized_nodes)
            #t_imm, S_imm, I_imm = EoN.basic_discrete_SIS(immu_G, 0.05, rho=0.85, tmax=20)
            #t_imm, S_imm, I_imm = EoN.basic_discrete_SIS(immu_G, 0.05, rho=0.95, tmax=20)
            if args.model == 'SIS':
                t_imm, S_imm, I_imm = EoN.basic_discrete_SIS(immu_G, p=args.trans_rate, rho=args.init_infecteds,
                                                             tmax=100)
            elif args.model == 'SIR':
                H = nx.DiGraph()
                H.add_node('S')
                #H.add_edge('E', 'I', rate=args.entrance_to_infected_rate)
                H.add_edge('I', 'R', rate=args.recover_rate)

                J = nx.DiGraph()
                J.add_edge(('I', 'S'), ('I', 'I'), rate=args.trans_rate)
                IC = defaultdict(lambda: 'S')
                sample_number = int(immu_G.number_of_nodes() * args.init_infecteds)
                initial_infected_nodes = random.sample(list(immu_G.nodes()), sample_number)
                for node in initial_infected_nodes:
                    IC[node] = 'I'

                return_statuses = ('S', 'I', 'R')

                t_imm, S_imm, I_imm, _ = EoN.Gillespie_simple_contagion(immu_G, H, J, IC, return_statuses,
                                                                               tmax=100)
            elif args.model == 'SEIR':
                H = nx.DiGraph()
                H.add_node('S')
                H.add_edge('E', 'I', rate=args.entrance_to_infected_rate)
                H.add_edge('I', 'R', rate=args.recover_rate)

                J = nx.DiGraph()
                J.add_edge(('I', 'S'), ('I', 'E'), rate=args.trans_rate)
                IC = defaultdict(lambda: 'S')
                sample_number = int(immu_G.number_of_nodes() * args.init_infecteds)
                initial_infected_nodes = random.sample(list(immu_G.nodes()), sample_number)
                for node in initial_infected_nodes:
                    IC[node] = 'I'

                return_statuses = ('S', 'E', 'I', 'R')

                t_imm, S_imm, E_imm, I_imm, _ = EoN.Gillespie_simple_contagion(immu_G, H, J, IC, return_statuses,
                                                                               tmax=100)
            #t_dict[method] = t_imm
            #I_dict[method] = I_imm
            print(I_imm)
            # t_imm, I_imm = t_imm[-20:], I_imm[-20:]
            if args.model == 'SIS':
                t_imm, I_imm = t_imm[1:], I_imm[1:]
            else:
                t_imm, I_imm = t_imm[-20:], I_imm[-20:]
            new_t_imm, new_I_imm = t_imm[::3], I_imm[::3]
            t_imm = np.concatenate((new_t_imm, np.array([t_imm[-1]])))
            I_imm = np.concatenate((new_I_imm, np.array([I_imm[-1]])))

            # # Save to a dataframe
            # df = pd.DataFrame({
            #     't_imm': t_imm,
            #     'I_imm': I_imm
            # })
            #
            # # Save to a CSV file
            # csv_path = "./epi_exp_data/"+args.model+"_"+args.dataset+"_"+str(args.k)+".csv"
            # df.to_csv(csv_path, index=False)
            with open("./epi_exp_data/"+args.model+"_"+args.dataset+"_"+str(args.k)+".txt", "w") as file:
                # Write headers
                file.write("t_imm\tI_imm\n")
                for t, I in zip(t_imm, I_imm):
                    file.write(f"{t}\t{I}\n")
            print(I_imm)
            colors = ['#D57DBF', '#7C534A', '#8D69B8', '#C53A32', '#519E3E', '#3B75AF', '#EF8636', '#28cc51', '#28aecc']
            if method == 'Hits':
                ax.plot(t_imm, I_imm, label='HITS', marker='^', markersize=10, color=colors[0])
            elif method == 'DINO':
                ax.plot(t_imm, I_imm, label='DINO', marker='D', markersize=10, color=colors[1])
            elif method == 'DegreeDirect':
                ax.plot(t_imm, I_imm, label='Degree',  marker='s', markersize=10, color=colors[2])
            elif method == 'Random':
                ax.plot(t_imm, I_imm, label='Random', marker='p', markersize=10, color=colors[3])
            elif method == 'PageRank':
                ax.plot(t_imm, I_imm, label=method,  marker='o', markersize=10, color=colors[4])
            elif method == 'KatzCentrality':
                ax.plot(t_imm, I_imm, label=method,  marker='v', markersize=10, color=colors[5])
            elif method == 'ClosenessCentrality':
                ax.plot(t_imm, I_imm, label=method,  marker='<', markersize=10, color=colors[6])
            elif method == 'Acquaintance':
                ax.plot(t_imm, I_imm, label=method,  marker='x', markersize=10, color=colors[7])
            elif method == 'CBF':
                ax.plot(t_imm, I_imm, label=method, marker='>', markersize=10, color=colors[8])
        output.append({"method": method, "t": t_imm.tolist(), "I":I_imm.tolist()})
#np.savez('EpiMod'+args.model+'_'
#            + args.dataset+'_'
#            +'trans_rate_'+str(args.trans_rate)
#            +'init_infe'+str(args.init_infecteds)
#            +'recov_rate'+str(args.recover_rate)
#            +'entran_infec_rate'+str(args.entrance_to_infected_rate)
#            +'.npz', t=t_dict,I=I_dict)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color="grey")
plt.gca().set_facecolor("#eeeeee")
plt.tight_layout()
ax.set_title(args.model, fontname='Times New Roman', fontsize=30)
ax.set_xlabel("time", fontname='Times New Roman', fontsize=28)
ax.set_ylabel("Infected people", fontname='Times New Roman', fontsize=35)

# legend = ax.legend(ncol=1, fontsize=25, frameon=False)  # Adjust bbox_to_anchor's y-coordinate (0.9) as needed
# for handle in legend.legendHandles:
#     handle.set_markersize(20)

plt.savefig('EpiMod'+args.model+'_'+args.dataset+'_'+'trans_rate_'+str(args.trans_rate)+'init_infe'+str(args.init_infecteds)+'recov_rate'+str(args.recover_rate)+'entran_infec_rate'+str(args.entrance_to_infected_rate)+'.pdf', dpi=300, bbox_inches='tight')
plt.show()
print(output)



# plt.grid(True, which='both', linestyle='--', linewidth=0.5, color="grey")
# plt.gca().set_facecolor("#eeeeee")
# plt.tight_layout()
# ax.set_title(args.model,fontname='Times New Roman', fontsize=30)
#
# ax.set_xlabel("time",fontname='Times New Roman', fontsize=28)
# ax.set_ylabel("Infected people",fontname='Times New Roman', fontsize=35)
#
#
# legend = ax.legend(ncol=1,  fontsize=25)
# for handle in legend.legendHandles:
#     handle.set_markersize(20)
# plt.savefig('EpiMod'+args.model+'_'
#             + args.dataset+'_'
#             +'trans_rate_'+str(args.trans_rate)
#             +'init_infe'+str(args.init_infecteds)
#             +'recov_rate'+str(args.recover_rate)
#             +'entran_infec_rate'+str(args.entrance_to_infected_rate)
#             +'.pdf', dpi=300)
# plt.show()