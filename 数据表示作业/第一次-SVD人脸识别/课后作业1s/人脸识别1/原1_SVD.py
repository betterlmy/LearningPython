# -*- coding: utf-8 -*-
"""
Created on 2019/3/25

@author: Sgh
"""
import numpy as np
import cv2


def readImg(path):
    Img = list()
    for i in range(40):
        for j in range(10):
            subPath = path + str(i + 1) + "/" + str(j + 1) + ".pgm"
            tmp = cv2.imread(subPath)
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            tmp = tmp.reshape((-1,))
            Img.append(tmp)
    return np.array(Img)


path = "./data/att_faces/s"

Img = readImg(path)
Train = Img[0::2]  # 序号为偶数的图片作为训练集
Test = Img[1::2]  # 序号为奇数的图片作为训练集
ImgFloat = Train.astype(np.float32)
ImgM = ImgFloat
ImgMean = Train.mean(axis=1)

for i in range(200):
    # 把每张照片所有元素都减去均值
    ImgM[i, :] = ImgFloat[i, :] - ImgMean[i]

sigma = np.dot(ImgM, ImgM.T)
v, d = np.linalg.eig(sigma)


def eigValPct(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 使用numpy中的sort()对特征值按照从小到大排序
    sortArray = sortArray[::-1]  # 特征值从大到小排序
    arraySum = np.sum(sortArray)  # 数据全部的方差arraySum
    tempSum = 0
    num = 0
    for i in sortArray:
        tempSum += i
        num += 1
        if tempSum >= arraySum * percentage:
            return num


def Acc(TestImg, FeaImg):
    cnt = 0
    for j in range(200):
        mdist = np.zeros((200,))
        for i in range(200):
            mdist[i] = np.linalg.norm(TestImg[j, :] - FeaImg[i, :])
        #    dist = np.sort(mdist)
        Ind = np.argmin(mdist)
        I = np.floor(Ind / 5) + 1
        l = np.floor(j / 5) + 1
        if I == l:
            cnt += 1
    return cnt / 200


k = eigValPct(v, 0.9)
eigValInd = np.argsort(v)
eigValInd = eigValInd[:-(k + 1):-1]
redEigVec = d[:, eigValInd]

base = np.dot(ImgM.T, redEigVec).dot(np.diag(np.power(v[:k], -0.5)))
FeaImg = np.dot(ImgFloat, base)

TestImg = np.dot(Test, base)
acc = Acc(TestImg, FeaImg)

print('The accuracy is %.2f%%' % (acc * 100))
