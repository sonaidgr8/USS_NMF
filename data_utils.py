from __future__ import division
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise
import networkx as nx


def binarize_adj_mat(adj_mat):
    adj_mat = np.where(adj_mat < 1, 0., 1.)
    return adj_mat


def make_undirected(adj_mat):
    pos = np.where(adj_mat > 0)
    adj_mat = adj_mat.astype(float)
    for (i, j) in zip(pos[0], pos[1]):
        val = (adj_mat[i, j] + adj_mat[j, i]) / 2.0
        adj_mat[i, j] = val
        adj_mat[j, i] = val
    return adj_mat


def get_all_label_distribution(truth):
    all_label_dist = np.true_divide(np.sum(truth, axis=0), np.sum(truth))
    return all_label_dist


def get_proximity_matrix(X, eta, nodelist=None):
    G = nx.from_numpy_matrix(X)
    if nodelist is None:
        nodelist = G.nodes()
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, format='csr')
    n, m = M.shape
    DI = np.diagflat(1.0 / np.sum(M, axis=1))
    P = DI * M
    A = (P + eta * np.dot(P, P)) / 2
    return np.array(A).reshape((n, n))


def get_proximity_similarity_matrix(X, eta):
    S = (X + eta * pairwise.cosine_similarity(X)) / 2
    S = normalize(S, axis=1, norm='l2')
    return S


def get_modularity_matrix(X, nodelist=None):
    G = nx.from_numpy_matrix(X)
    n, m = X.shape
    if nodelist is None:
        nodelist = G.nodes()
    if nx.is_directed(G):
        M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, format='csr')
        k_in = M.sum(axis=0)
        k_out = M.sum(axis=1)
        m = G.number_of_edges()
        B = k_out * k_in / m
    else:
        M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, format='csr')
        k = M.sum(axis=1)
        m = G.number_of_edges()
        B = k * k.transpose() / (2 * m)
    return np.array(B).reshape((n, n))
