#coding=utf-8

import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
# 转化图片为固定尺寸

# root = "E:\Artificial intelligence\SVD_Face_Detection\Dev\\"
root = "E:\Artificial intelligence\SVD_Face_Detection\\att_faces/"
dsize = 100
persons = os.listdir(root)
X = np.zeros((400, 100, 100, 3))
y = np.zeros((400, 1))

index = 0
for i in tqdm(range(len(persons))):
    images = os.listdir(os.path.join(root, persons[i]))
    for img in images:
        X[index] = cv2.resize(cv2.imread(os.path.join(root, persons[i], img)), (dsize,dsize))
        # plt.imshow(X[index].astype("int"))
        # plt.show()
        y[index] = i
        index += 1

np.save("train.npy", X)
np.save("train_label.npy", y)
