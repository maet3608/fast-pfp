"""
.. module:: match
   :synopsis: Implements the graph matching algorithm described in
              https://arxiv.org/pdf/1207.1114
              using numpy and Pytorch
"""
from __future__ import print_function

import numpy as np

from fastpfp.util import datatype
from torch import mm, zeros, ones, eye, Tensor


def loss(A1, A2, L1, L2, X, lam=0.0):
    """
    Return loss for graph matching.

    :param torch.Tensor A1: Adjacency matrix of first graph.
    :param torch.Tensor A2: Adjacency matrix of second graph.
    :param torch.Tensor L1: Node vector (=labels) matrix of first graph.
    :param torch.Tensor L2: Node vector (=labels) matrix of second graph.
    :param torch.Tensor X: Result matrix of pfp()
    :param float lam: Trade off between matching of node vectors and
       matching of edges.
    :return: loss
    :rtype: float
    """
    edge_loss = 0.5 * (A1 - mm(X, mm(A2, X.t()))).norm()
    node_loss = (L1 - mm(X, L2)).norm()
    loss = (edge_loss + lam * node_loss)
    return loss.numpy()


def discretize(X):
    """
    Returns partial permutation matrix for result PFP result matrix X.

    :param np.array X: Result matrix return by pfp()
    :return: Partial permutation matrix
    :rtype: np.array
    """
    X = X.copy()
    min_x = X.min() - 1.
    P = np.zeros(X.shape)
    while (X > min_x).any():
        r, c = np.unravel_index(X.argmax(), X.shape)
        P[r, c] = 1.
        X[r, :] = min_x
        X[:, c] = min_x
    return P


def num_nodes(A1, A2):
    """Return number of nodes for the two adjacency matrices given.

    Also checks that adjacency matrices are square, symmetric and that the
    number of nodes (rows,cols) of A1 is greater or equal to A2
    (as required by the algorithm).
    Adjacency matrices can be binary or float matrices, e.g.
    distances between graph nodes.

    :param np.array A1: First adjacency matrix.
    :param np.array A2: Second adjacency matrix.
    :rtype : tuple(int, int)
    :return: Number of rows (=cols) of matrix A1 and matrix A2
    """
    n1, m1 = A1.shape
    n2, m2 = A2.shape
    assert n1 == m1, 'A1 must be square!'
    assert n2 == m2, 'A2 must be square!'
    assert n1 >= n1, 'number of rows/cols in A1 >= A2 is required'
    assert np.allclose(A1, A1.T, atol=1e-8), 'A1 must be symmetric!'
    assert np.allclose(A2, A2.T, atol=1e-8), 'A2 must be symmetric!'
    return n1, n2


def pfp(A1, A2, L1, L2, alpha=0.5, lam=1.0, device_id=None, verbose=False):
    """
    Matches two graphs given by node vectors and adjacency matrices.

    Note that adjacency matrices must be square, symmetric and that the
    number of nodes (rows,cols) of A1 is greater or equal to A2.
    Adjacency matrices can be binary or float matrices, e.g.
    distances between graph nodes.

    :param np.array A1: Adjacency matrix of first graph.
    :param np.array A2: Adjacency matrix of second graph.
    :param np.array L1: Node vector (=labels) matrix of first graph.
    :param np.array L2: Node vector (=labels) matrix of second graph.
    :param float alpha: Step size.
    :param float lam: Trade off between matching of node vectors and
       matching of edges.
    :param int|None device_id:
       None: automatic. Pick GPU if available otherwise CPU.
      -1: CPU
      int: Device id.
    :param bool verbose: Print loss if True.
    :return: result matrix of projected fixed point method
    :rtype: np.array
    """
    threshold1 = threshold2 = 1.0e-6
    max_iter1 = max_iter2 = 100
    dt = datatype(device_id)
    n1, n2 = num_nodes(A1, A2)

    L1 = Tensor(L1).type(dt)
    L2 = Tensor(L2).type(dt)
    A1 = Tensor(A1).type(dt)
    A2 = Tensor(A2).type(dt)

    o1 = ones(n1, 1).type(dt)
    O1 = ones(n1, n2).type(dt)
    O2 = ones(n1, n1).type(dt)
    I = eye(n1).type(dt)

    X = O1.div(n1 * n2).type(dt)
    Y = zeros(n1, n1).type(dt)
    K = mm(L1, L2.t()).type(dt)
    I1 = I / n1

    for _ in range(max_iter1):
        Y[:n1, :n2] = mm(A1, mm(X, A2)) + lam * K
        for _ in range(max_iter2):
            T = I1 + (mm(o1.t(), mm(Y, o1)) / (n1 * n1)) * I - Y / n1
            Ynew = Y + mm(T, O2) - mm(O2, Y) / n1
            Ynew = (Ynew + Ynew.abs()) / 2
            eps2 = (Ynew - Y).abs().max()
            Y = Ynew
            if eps2 < threshold2:
                break
        Xnew = (1 - alpha) * X + alpha * Y[:n1, :n2]
        Xnew /= Xnew.max()
        eps1 = (Xnew - X).abs().max()
        X = Xnew
        if verbose:
            print('loss =', loss(A1, A2, L1, L2, X))
        if eps1 < threshold1:
            break
    return X.numpy()


def match_graphs(*args, **kwargs):
    """
    Returns node permuation matrix for matched graphs.

    :param args: See pfp()
    :param kwargs: See pfp()
    :return: Permutation matrix
    :rtype: np.array
    """
    return discretize(pfp(*args, **kwargs))
