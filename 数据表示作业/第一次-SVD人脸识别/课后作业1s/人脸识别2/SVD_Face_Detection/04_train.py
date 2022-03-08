#coding=utf-8
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

train = np.load("./train.npy")
y = np.load("./train_label.npy").reshape(-1, )

# 对每个样本使用奇异值分解并组合成向量
# k = 10    # 82
# k = 5     # 82
k = 20
n = 400

def transfer(train):
    x = np.zeros((n, k*100*2+k))
    print(x.shape)

    for index in tqdm(range(train.shape[0])):
        # 图片转化为灰度图
        # dst = 0.2989 * train[index,:,:,0] + 0.5870 * train[index,:,:,1] + 0.1140 * train[index,:,:,2]
        dst = 0.1140 * train[index,:,:,0] + 0.5870 * train[index,:,:,1] + 0.2989 * train[index,:,:,2]
        # 图片进行奇异值分解
        u, lamda, v = np.linalg.svd(dst)
    
        # 堆叠为向量的形式
        vector = u[:k].flatten().tolist() + v[:k].flatten().tolist() + lamda[:k].flatten().tolist()
        x[index, :] = vector        

    return x    

# 获取转化为向量的训练数据
x = transfer(train)
x, y = shuffle(x, y)


skf = StratifiedKFold(n_splits=5)
# skf.get_n_splits()
for train_index, test_index in skf.split(x, y):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # svm linear
    svc = SVC(gamma="auto", kernel="linear", C=100)
    svc.fit(X_train, y_train)
    svc_pred = svc.predict(X_test)

    # rf
    rf = RandomForestClassifier(n_estimators=1000) # 0.825
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print("Fold SVM, RF acc: %0.2f, %0.2f" % (accuracy_score(y_test, svc_pred), accuracy_score(y_test, rf_pred)))

