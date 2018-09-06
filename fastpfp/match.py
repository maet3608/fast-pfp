"""
Implements the graph matching algorithm described in
https://arxiv.org/pdf/1207.1114
using numpy and Pytorch
"""
from __future__ import print_function

import torch.cuda as cuda
import numpy as np

from torch import mm, zeros, ones, eye, Tensor, FloatTensor


def check_gpu():
    print(cuda.is_available())
    print(cuda.get_device_name(0))


def loss(A1, A2, L1, L2, X, lam=0.0):
    return 0.5 * (A1 - mm(X, mm(A2, X.t()))).norm() + lam * (
            L1 - mm(X, L2)).norm()


def discretize(X):
    X = X.numpy().copy()
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


def datatype(device_id):
    if device_id is None:
        dt = FloatTensor
    else:
        dt = cuda.FloatTensor
        cuda.set_device(device_id)
    return dt


def pfp(A1, A2, L1, L2, alpha=0.5, lam=1.0, device_id=None):
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
        # print('loss =', loss(A1, A2, L1, L2, X))
        if eps1 < threshold1:
            break
    return X


def run():
    L1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    L2 = np.array([[0, 0], [0, 1], [1, 0]])
    A1 = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]])
    A2 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])

    X = pfp(A1, A2, L1, L2, lam=1.0, device_id=None)
    P = discretize(X)

    R = P.dot(A2.dot(P.T))
    D = A1 - R
    print('*******************')
    print(P)
    print(X)
    print(A1)
    print(R)
    print(D)


if __name__ == '__main__':
    # check_gpu()
    run()
