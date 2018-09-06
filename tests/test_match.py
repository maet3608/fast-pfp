"""
.. module:: test_fastpfp
   :synopsis: Unit tests for fastpfp module
"""

import numpy as np

from fastpfp.match import num_nodes, discretize, pfp


def create_test_data():
    L1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    L2 = np.array([[0, 0], [0, 1], [1, 0]])
    A1 = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]])
    A2 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])

    return L1, L2, A1, A2


def test_num_nodes():
    _, _, A1, A2 = create_test_data()
    n1, n2 = num_nodes(A1, A2)
    assert n1 == 4
    assert n2 == 3


def test_match():
    L1, L2, A1, A2 = create_test_data()

    X = pfp(A1, A2, L1, L2, lam=1.0, device_id=None)
    P = discretize(X)

    P_expected = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
    assert np.allclose(P, P_expected)
