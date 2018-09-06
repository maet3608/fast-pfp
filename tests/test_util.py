"""
.. module:: test_util
   :synopsis: Unit tests for util module
"""

import numpy as np

from fastpfp.util import graph2matrices, adj_matrix2edges


def test_graph2matrices():
    L, A = graph2matrices(4, [(0, 1), (0, 2)])

    L_expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    A_expected = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])

    assert np.allclose(L, L_expected)
    assert np.allclose(A, A_expected)


def test_graph2matrices_weighted():
    labels = [[1], [2], [3]]
    edges = [(0, 1, 0.1), (0, 2, 0.2)]
    L, A = graph2matrices(labels, edges)

    L_expected = np.array([[1], [2], [3]])
    A_expected = np.array([[0, 0.1, 0.2], [0.1, 0, 0], [0.2, 0, 0]])

    assert np.allclose(L, L_expected)
    assert np.allclose(A, A_expected)


def test_adj_matrix2edges():
    edges = [(0, 1, 0.1), (0, 2, 0.2)]
    _, A = graph2matrices(3, edges)

    assert adj_matrix2edges(A) == edges
