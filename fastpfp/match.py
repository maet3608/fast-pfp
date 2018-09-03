"""
Implements the graph matching algorithm described in
https://arxiv.org/pdf/1207.1114
"""
from __future__ import print_function

import torch
import numpy as np


def check_gpu():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))


def pfp(A1, A2, dt):
    alpha = 0.5
    lam = 10
    n1, m1 = A1.shape
    n2, m2 = A2.shape
    assert n1 == m1
    assert n2 == m2
    assert n1 >= n1
    print('A1', A1.shape)
    print('A2', A2.shape)

    o1 = torch.ones(n1, 1).type(dt)

    O1 = torch.ones(n1, n2).type(dt)
    print('O1', O1.shape)
    O2 = torch.ones(n1, n1).type(dt)
    print('O2', O2.shape)
    I = torch.eye(n1).type(dt)
    print('I', I.shape)

    X = O1.div(n1 * n2).type(dt)
    print('X', X.shape)
    Y = torch.zeros(n1, n1).type(dt)
    print('Y', Y.shape)
    K = torch.zeros(n1, n2).type(dt)
    print('K', K.shape)

    dot = torch.mm

    n_i, n_j = 5, 5
    for i in range(n_i):
        print('i', i, '-----------------------------')
        Y[:n1, :n2] = dot(A1, dot(X, A2)) + K.mul(lam)
        print('  Y', Y.shape)
        for j in range(n_j):
            print('j', j, '---------------')
            T = I / n1
            T = T + dot(o1.t(), dot(Y, o1)).div(n1 * n1) * I
            T = T - Y / n1
            Y = Y + dot(T, O2) - dot(O2 /n1, Y)
            #print(Y)
        X = (1 - alpha) * X + alpha * Y[:n1, :n2]
        X = X / X.max()
        print(X)


def run():
    # dt = torch.cuda.FloatTensor if torch.cuda.is_available() else
    # torch.FloatTensor
    dt = torch.FloatTensor

    # square adjacency matrices with shape of A1 >= A2
    # must be symmetric by can be weighted
    A1 = torch.Tensor([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]])
    A2 = torch.Tensor([[0, 1, 1], [1, 0, 0], [1, 0, 0]])

    pfp(A1, A2, dt)


if __name__ == '__main__':
    check_gpu()
    run()

