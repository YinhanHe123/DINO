import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time
import sys
import math
sys.path.append('../methods/')

from measures.CONTAIN import contain_alg
from measures.DegreeIterative import degree_iterative_alg

# from methods.PageRank import pagerank_alg
from measures.DegreeDirect import degree_direct_alg
from measures.Shi import shi_alg
from measures.Random import random_alg
# from methods.Walk import walk_alg
from measures.BruteForce import brute_force_alg
from measures.NewWalk import new_walk_alg
from measures.Ext_perception import ext_perception_alg, ite_ext_perception_alg
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
from measures.DirectedEigenPerturb import directed_eigen_perturb_alg
import scipy.sparse as sp
import argparse

def plot_node_time():
    # Data parsing
    # Generating the complete data_dict for all provided data
    def execute_alg(method, A, args):
        if args.k != None:
            k = args.k
        else:
            k = int(args.node_selection_rate * A.shape[0])
        if method == 'Centrality':
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            immunized_nodes_list, run_times = centrality_alg(G, k)

            Lam_A_immunized_list = []
            for immunized_nodes in immunized_nodes_list:
                G_ = G.copy()
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
                # G = nx.from_numpy_array(A, create_using=nx.DiGraph)
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
                immunized_nodes = brute_force_alg(A, k)
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
                                  'eigenvector_centrality',
                                  'closeness_centrality',
                                  'betweenness_centrality']:
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
                immunized_nodes = katz_centrality_alg(G, k)
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
                print('time for finding immunization nodes by new_SCC:', end_finding - start_finding)

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
            else:
                Lam_A_immunized = []
                cent_mode_list = ['Random', 'PageRank', 'DegreeDirect', 'Hits', 'ext_perception',
                                  'eigenvector_centrality',
                                  'closeness_centrality',
                                  'betweenness_centrality']
                for i, nodes in enumerate(immunized_nodes):
                    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
                    G.remove_nodes_from(nodes)
                    A_subgraph = nx.adjacency_matrix(G)
                    try:
                        immu_lam, _ = sp.linalg.eigs(A_subgraph, 1, which='LR')
                        immu_lam = abs(immu_lam).real
                        print("The immunized spectrum radius of SCC-", cent_mode_list[i], "is ", immu_lam)
                    except:
                        immu_lam = 0
                    Lam_A_immunized.append(immu_lam)
            return Lam_A_immunized, run_times

    def generate_power_law_sequence(n, exponent):
        # Generate a sequence of integers that follows a power-law distribution
        np.random.seed(33)
        s = (np.random.zipf(exponent, n) - 1) * 5
        return s

    def power_law_directed_graph(n, exponent):
        # Generate in-degree and out-degree sequences
        in_degrees = generate_power_law_sequence(n, exponent)
        D = nx.directed_configuration_model(in_degrees, in_degrees, create_using=nx.DiGraph(), seed=33)
        # Remove self-loops and parallel edges
        D = nx.DiGraph(D)
        D.remove_edges_from(nx.selfloop_edges(D))
        return D


    nodes_list = [100, 200, 500, 1000, 2000, 3500, 5000, 6500, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    elements_count = [264, 678, 1996, 4242, 9213, 16429, 24643, 31764, 38000, 50491, 60498, 70630, 81395, 92465,
                      101544]
    methods_list = ['DINO']
    table_value = np.zeros((len(methods_list), len(nodes_list)))
    table_var = np.zeros((len(methods_list), len(nodes_list)))
    data_dict = {264: {}, 678: {}, 1996: {}, 4242: {}, 9213: {}, 16429: {}, 24643: {}, 31764: {},
                 38000: {}, 50491: {}, 60498: {}, 70630: {}, 81395: {}, 92465: {}, 101544: {}}
    var_dict = {264: {}, 678: {}, 1996: {}, 4242: {}, 9213: {}, 16429: {}, 24643: {}, 31764: {},
                38000: {}, 50491: {}, 60498: {}, 70630: {}, 81395: {}, 92465: {}, 101544: {}}
    for i, node_num in enumerate(nodes_list):
        Graph = power_law_directed_graph(node_num, 2.5)
        A = nx.adjacency_matrix(Graph).asfptype()
        for j, method in enumerate(methods_list):
            time_list = []
            for k in range(2):
                if method == 'DINO':
                    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
                    start_finding = time.time()
                    immunized_nodes = scc_alg(G, 20, 'dino', None)
                    end_finding = time.time()
                    run_times = end_finding - start_finding
                if method == 'ExPSCC':
                    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
                    start_finding = time.time()
                    immunized_nodes = new_scc_alg(G, 20, 'ext_Perception', 1, None)
                    end_finding = time.time()
                    run_times = end_finding - start_finding
                if method == 'ExtPerception':
                    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
                    start_finding = time.time()
                    immunized_nodes = ext_perception_alg(G, 20)
                    end_finding = time.time()
                    run_times = end_finding - start_finding
                    print('time for finding immunization nodes by ext_perception:', end_finding - start_finding)
                time_list.append(end_finding - start_finding)
            print(np.mean(time_list), np.var(time_list))
            table_value[j][i] = np.mean(time_list)
            table_var[j][i] = np.var(time_list)
            data_dict[elements_count[i]][method] = np.mean(time_list)
            var_dict[elements_count[i]][method] = math.sqrt(np.var(time_list))

    # Now perform three methods for 5 times each, calculate their time and variance.

    # Plot settings
    plt.rcParams['font.family'] = 'Times New Roman'
    # colors = plt.cm.tab20(np.linspace(0, 1, len(methods_list)))
    colors = ["#367E21", "#AD5BCD", "#B6D7E4"]    

    plt.figure(figsize=(8, 6))
    for idx, method in enumerate(methods_list):
        x_vals = list(data_dict.keys())
        y_vals = [data_dict[x].get(method, 0) for x in x_vals]
        y_error = [var_dict[x].get(method, 0) for x in x_vals]
        upper_bound = [y + e for y, e in zip(y_vals, y_error)]
        lower_bound = [y - e for y, e in zip(y_vals, y_error)]
        print(x_vals)
        print(y_vals)
        print(y_error)
        print(upper_bound)
        print(lower_bound)
        marker = "D" if method == "ExPSCC" else "^"
        plt.plot(x_vals, y_vals, label=method, marker=marker, color=colors[idx], markersize=15)
        # Adding the shaded region here
        plt.fill_between(x_vals, lower_bound, upper_bound, color=colors[idx], alpha=0.3)

    # plt.title("Running time for Various Methods", fontsize=16)
    plt.xlabel("|V|+|E|", fontsize=14)
    plt.ylabel("Running Time", fontsize=14)
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color="grey")
    plt.gca().set_facecolor("#eeeeee")
    plt.tight_layout()
    # plt.savefig("node_time_plot.pdf")
    # plt.show()

def scale_plot():
    # Data
    labels = ["Eigenvec.Cen.", "Close.Cen", "Betw.Cen.", "ExPSCC", "Acquaintance", "CBF", "Random", "PageRank",
              "Degree", "Hits"]
    spectral_radius_drop = [1.4853, 1.6244, 1.7159, 1.8612, 0.0685, 1.4677, 0.0134, 1.4874, 1.6388, 0.0953]
    running_time = [0.0855, 9.2186, 44.0978, 0.0052, 0.3718, 0.0003, 0.0034, 0.0016, 1.8743]
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'lime']

    # Plotting Spectral Radius Drop vs Running Time with specific marker sizes
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        if i < len(running_time):
            plt.scatter(running_time[i], spectral_radius_drop[i], marker=markers[i], s=500, color=colors[i],
                        label=label if i == 0 else "")  # Only label the first to control legend size

    # Setting legend with custom handle for marker size
    handles, labels = plt.gca().get_legend_handles_labels()
    legend = plt.legend(handles[:1], labels[:1], loc="best", markerscale=0.45)

    plt.title("Spectral Radius Drop vs Running Time")
    plt.ylabel("Spectral Radius Drop")
    plt.xlabel("Running Time (s)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="grey")
    plt.gca().set_facecolor('#eeeeee')
    plt.tight_layout()

# Combining the plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

plt.sca(axes[0])
plot_node_time()
# plt.title("(a)")

plt.sca(axes[1])
scale_plot()
# plt.title("(b)")

plt.tight_layout()

plt.savefig("combined_time_plot.pdf")
plt.show()