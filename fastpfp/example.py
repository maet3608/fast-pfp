"""
.. module:: example
   :synopsis: A usage example
"""
from __future__ import print_function
from fastpfp.util import graph2matrices
from fastpfp.match import match_graphs

if __name__ == '__main__':
    L1, A1 = graph2matrices([[0], [1], [2], [3]], [(0, 1), (0, 2), (2, 3)])
    L2, A2 = graph2matrices([[1], [0], [2]], [(1, 0), (1, 2)])

    print('first graph -----------------------------------------')
    print(L1)
    print(A1)
    print('second graph ----------------------------------------')
    print(L2)
    print(A2)

    P = match_graphs(A1, A2, L1, L2, lam=1.0, device_id=-1)
    print('permutation matrix----------------------------------------')
    print(P)
