"""
Utility function
"""

from numpy import np


def edges2matrix(nodes, edges):
    n = len(nodes)
    L = np.array([v for n, v in nodes])
    A = np.zeros((n, n))
    idx = {n: i for i, (n, v) in enumerate(nodes)}
    for node1, node2, weight in edges:
        i, j = idx[node1], idx[node2]
        A[i, j] = A[j, i] = weight
    return A, L


def matrix2edges(nodes, A):
    edges = []
    rs, cs = np.where(A > 0.5)
    for r, c in zip(rs, cs):
        if c > r:
            edges.append((nodes[r][0], nodes[c][0], A[r, c]))
    return edges


if __name__ == '__main__':
    nodes = (
        ('A', [0, 0]),
        ('B', [0, 1]),
        ('C', [1, 0]),
        ('D', [1, 1]),
    )
    edges = (
        ('A', 'B', 1),
        ('A', 'C', 1),
        ('D', 'C', 1),
    )
    A, L = edges2matrix(nodes, edges)
    print(A)
    print(L)

    edges = matrix2edges(nodes, A)
    for edge in edges:
        print(edge)
