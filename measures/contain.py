import math
import scipy.sparse as sp
import numpy as np

from measures.base_measure import BaseMeasure

class Contain(BaseMeasure):
    def __init__(self, *args):
        self.r = args.r
        super().__init__("CONTAIN")

    def update_eigen_decomp(self,old_vecs, old_vals, A, selected_node):
        neighbors_out = A[[selected_node],:].nonzero()[1]
        neighbors_in = A[:, [selected_node]].nonzero()[0]
        neighbors_intersect = np.array(list(set(neighbors_in).intersection(set(neighbors_out))))
        num_intersect = len(neighbors_intersect)
        if num_intersect == 0:
            return old_vecs, old_vals
        Y = np.diag([math.sqrt(num_intersect), -math.sqrt(num_intersect)])
        X = np.zeros((len(A), 2))
        X[selected_node, 0] = 1 / math.sqrt(2)
        X[selected_node, 1] = 1 / math.sqrt(2)
        X[neighbors_intersect, 0] = -1 / math.sqrt(2 * num_intersect)
        X[neighbors_intersect, 1] = 1 / math.sqrt(2 * num_intersect)
        Q, R = self.get_Q_R(old_vecs, X, selected_node, neighbors_intersect)
        M = np.diag(np.concatenate((old_vals, np.diag(Y)), axis=0))
        Z = np.matmul(np.matmul(R, M), np.transpose(R))
        r = np.linalg.matrix_rank(Z)
        if r < len(Z):
            new_vals, V = sp.linalg.eigs(Z, k=r, which='LM')
        else:
            new_vals, V = sp.linalg.eigs(Z)
        top_r = min(len(old_vecs[0]), len(new_vals))
        new_vals = new_vals[:top_r]
        new_vecs = np.matmul(Q, V)[:, :top_r]
        return new_vecs, new_vals
    
    def get_Q_R(self, eigen_vecs, X, selected_node, neighbors_intersect):
        num_intersect = len(neighbors_intersect)
        r1 = 1 / math.sqrt(2) * eigen_vecs[selected_node, :] - 1 / math.sqrt(2 * num_intersect) * np.sum(eigen_vecs[neighbors_intersect, :], axis=0)
        r1 = np.expand_dims(r1, axis=0).transpose()
        norm_r1_square = np.sum(np.abs(r1)**2)
        r2 = 1 / math.sqrt(2) * eigen_vecs[selected_node, :] + 1 / math.sqrt(2 * num_intersect) * np.sum(eigen_vecs[neighbors_intersect, :], axis=0)
        r2 = np.expand_dims(r2, axis=0).transpose()
        norm_r2_square = np.sum(np.abs(r2)**2)
        q1_norm = math.sqrt(abs(1 - norm_r1_square))
        if q1_norm < 1e-8:
            q2_norm = math.sqrt(abs(1 - norm_r2_square))
            q2 = np.transpose(np.expand_dims(X[:, 1], 0)) - np.matmul(eigen_vecs, r2)
        else:
            q2_norm = math.sqrt(abs(1 - norm_r2_square -
                                    np.sum(np.abs(np.matmul(np.transpose(r1), r2)) ** 2 / (1-norm_r1_square))))
            q2 = np.transpose(np.expand_dims(X[:, 1], 0)) - np.matmul(eigen_vecs, r2) + q1 * np.matmul(np.transpose(r1), r2) / (
                        q1_norm * q1_norm)
        q1 = np.transpose(np.expand_dims(X[:, 0], 0)) - np.matmul(eigen_vecs, r1)
        Q = np.concatenate((eigen_vecs, q1 / q1_norm, q2 / q2_norm), axis=1)
        t = len(eigen_vecs[0])
        R = np.zeros((t + 2, t + 2), dtype='complex_')
        R[0:t] = np.concatenate((np.eye(t), r1, r2), axis=1)
        R[t] = np.concatenate((np.zeros(t), np.array([q1_norm,
                1 / q1_norm * np.matmul(np.transpose(r1), r2).item()])), axis=0)
        R[t + 1] = np.concatenate((np.zeros(t + 1), np.array([q2_norm])), axis=0)
        if q1_norm < 1e-8:
            Q[:, len(Q[0]) - 2] = 0
            R[t, t:] = 0
        if q2_norm < 1e-8:
            Q[:, len(Q[0]) - 1] = 0
            R[t + 1, t + 1] = 0
        return Q, R
    
    def get_immunized_nodes(self, A, k):
        A_ = A.copy()
        e_vals, e_vecs = sp.linalg.eigs(A_, self.r, which='LM')
        nodes_to_select = np.ones(A_.shape[0])  # every time when selected one node, try to switch that one to 0.
        for _ in range(k):
            F = -1e10
            node_selected = -1
            for node in np.where(nodes_to_select == 1)[0]:
                _, new_e_vals = self.update_eigen_decomp(e_vecs, e_vals, A_, node)
                F_new = e_vals[0] - new_e_vals[0]
                if F_new > F:
                    node_selected = node
                    F = F_new
            if node_selected >= 0:
                nodes_to_select[node_selected] = 0
            e_vecs, e_vals = self.update_eigen_decomp(e_vecs, e_vals, A_, node_selected)
            A_[node_selected] = 0
            A_[:, node_selected] = 0
        return np.where(nodes_to_select == 0)[0].tolist()