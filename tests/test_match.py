"""
.. module:: test_fastpfp
   :synopsis: Unit tests for fastpfp module
"""

import numpy as np

from fastpfp.match import num_nodes

def test_num_nodes():
    A1 = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]])
    A2 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    n1, n2 = num_nodes(A1, A2)
    assert n1 == 4
    assert n2 == 3