# 实现svd分解
import math

import numpy as np


def all_negetive(vectors):
    # 判断这个向量是不是全负,如果是则返回True
    for vector in vectors:
        if vector > 0:
            return False
    return True


def check(featureVectors):
    featureVectors = featureVectors.T
    # 特征向量如果全负,则取反,按列操作
    for i, vectors in enumerate(featureVectors):
        if all_negetive(vectors):
            featureVectors[i] *= -1  # 取反操作
    return featureVectors


def getEigen(A, B):
    # 获取A@B的特征值和特征向量
    tmp = np.dot(A, B)
    eigenvalues, featureVectors = np.linalg.eig(tmp)
    featureVectors = check(featureVectors)  # 保证所有的向量不是全负
    eigenvalues = list(eigenvalues)
    featureVectors = list(featureVectors)
    i = 0
    valueVectorList = []
    for eigenvalue, featureVector in zip(eigenvalues, featureVectors):
        valueVectorList.append((i, eigenvalue, featureVector))
        i += 1
    valueVectorList.sort(key=lambda x: x[1], reverse=True)

    return valueVectorList


def SVD(A, thin=False, diag=True):
    # 奇异值分解

    valueVectorList1 = getEigen(A, A.T)
    eigenvalue1 = []
    U = []
    for x in valueVectorList1:
        eigenvalue1.append(x[1])
        U.append(x[2])
    U = np.array(U)

    valueVectorList2 = getEigen(A.T, A)
    eigenvalue2 = []
    V = []
    for x in valueVectorList2:
        eigenvalue2.append(x[1])
        V.append(x[2])
    V = np.array(V)

    if len(eigenvalue1) > len(eigenvalue2):
        eigenvalue = eigenvalue2
    else:
        eigenvalue = eigenvalue1

    m, n = A.shape
    if thin and m > n:
        # U需要从m*n变为n*n
        m = n
        U = U.T[:n].T
    # 求E
    E = np.array(eigenvalue)
    if diag:
        E = np.zeros([m, n])
        for i, value in enumerate(eigenvalue):
            E[i, i] = math.sqrt(value)
    return U, E, V


if __name__ == '__main__':
    A1 = np.array([
        [1, 0, 0, 0, 2],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0],
    ])
    A2 = np.array([
        [2, 0],
        [0, 1 / math.sqrt(2)],
        [0, -1 / math.sqrt(2)],
    ])
    A = np.array([
        [0, 1],
        [1, 1],
        [1, 0],
    ])
    isThin = False
    U, E, V = SVD(A, isThin)

    B = np.dot(np.dot(U, E), V.T)
    if isThin:
        print("矩阵thinSVD分解得A=UEV,其中")
    else:
        print("矩阵奇异值分解得A=UEV,其中")
    print("U=", U)
    print("E=", E)
    print("V=", V)
    print(A == B)
