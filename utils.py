import argparse
from measures.measure_utils import MEASURE_MAP

def get_args():
    parser = argparse.ArgumentParser(description='DINO')
    parser.add_argument('-d', '--dataset', type=str, default='p2p-Gnutella08', choices=['hiv_transmission', 'Email-EuAll', 'p2p-Gnutella08', 'erdos_renyi', 'soc-Epinions1', 'WikiTalk', 'Wiki-Vote'],
                        help='the directed network dataset')
    parser.add_argument('-k', '--k', type=int, default=None, help='number of nodes for immunization')
    parser.add_argument('-r', '--r', type=int, default=2,
                        help='number of eigenvalues considered for recovering the adjacency matrix')
    parser.add_argument('-m', '--measure', type=str, default='dino', choices=list(MEASURE_MAP.keys()), help='node immunization algorithms')

    parser.add_argument('-nn', '--node_num', type=int, default=130, help='node number of the netowrk if args.dataset is erdos_renyi')

    parser.add_argument('-erd', '--erdos_renyi_density', type=float, default=0.01,
                        help='erdos renyi graph density, now not used in erdos_renyi_unon')
    parser.add_argument('-kp', '--k_percent', type= float, default=0.05,
                        help='the percent of nodes in dataset to be selected to immunize.')
    parser.add_argument('-et', '--epidemic_type', type=str, default=None, choices=['SIR', 'SIS', 'SEIR'],
                        help='epidemic type for epidemic simulation')
    return parser.parse_args()