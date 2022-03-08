# 实现svd分解
import math

import numpy as np
from numpy.linalg import svd

if __name__ == '__main__':
    A = np.array([
        [1, 0, 0, 0, 2],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0],
    ])
    # A = np.array([
    #     [2, 0],
    #     [0, 1 / math.sqrt(2)],
    #     [0, -1 / math.sqrt(2)],
    # ])
    # A = np.array([
    #     [0, 1],
    #     [1, 1],
    #     [1, 0],
    # ])

    U, eigenvalues, V = svd(A)
    m, n = A.shape
    E = np.zeros([m, n])
    for i, value in enumerate(eigenvalues):
        E[i, i] = math.sqrt(value)

    B = np.dot(np.dot(U, E), V)

    print("U=", U)
    print("E=", E)
    print("V=", V)
    print(A == B)
