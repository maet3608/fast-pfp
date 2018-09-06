"""
.. module:: util
   :synopsis: Utility functions
"""

import numpy as np
import torch.cuda as cuda

from torch import FloatTensor


def datatype(device_id):
    """
    Return tensor data type for CPU or GPU.

    :param int|None device_id: Device to create datatype for.
      None: automatic. Pick GPU if available otherwise CPU.
      -1: CPU
      int: Device id.
    :return: cuda.FloatTensor or FloatTensor
    """
    if device_id is None:
        return cuda.FloatTensor if cuda.is_available() else FloatTensor
    if device_id < 0:
        return FloatTensor
    cuda.set_device(device_id)
    return cuda.FloatTensor


def graph2matrices(labels, edges):
    """
    Return label and adjacency matrix for graph.

    :param int|list labels: list of vectors for each node of the graph
            or the number of label vectors to create.
    :param list edges: List of edges where each edge is described by a tuple
           containing the indices of the two graph node (and a edge weight).
    :return: Label matrix and adjacency matrix
    :rtype: tuple(np.array, np.array)
    """
    edges = np.array(edges)
    has_weights = edges.shape[1] > 2
    rs, cs = edges[:, 0].astype(int), edges[:, 1].astype(int)
    w = edges[:, 2] if has_weights else 1
    n = int(edges[:, :2].max()) + 1
    A = np.zeros((n, n))  # adjacency matrix
    A[rs, cs] = w
    A[cs, rs] = w
    L = np.eye(n, labels) if isinstance(labels, int) else np.array(labels)
    return L, A


def adj_matrix2edges(A):
    """
    Returns edge list for given adjacency matrix
    :param np.array A: square, symmetric adjacency matrix.
    :return: edge list with weight from adjacency matrix
    ;rtype: list of tuples of the form (int, int, float)
    """
    rs, cs = np.where(A != 0)
    return [(r, c, A[r,c]) for r, c in zip(rs, cs) if c > r]
