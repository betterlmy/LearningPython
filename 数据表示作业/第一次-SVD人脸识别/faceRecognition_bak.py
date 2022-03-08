import time

import cv2
import numpy as np

from svd import SVD


def img2array():
    # 将所有图片读取为ndarrayList
    path = "./att_faces/s"
    trainImgList = []

    for _ in range(1, 41):
        for __ in range(1, 10 + 1):
            imgPath = path + str(_) + "/" + str(__) + ".pgm"
            img = cv2.imread(imgPath)  # 通过opencv读取图片
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转为灰度图
            trainImgList.append(img)
    trainImgList = np.array(trainImgList)
    return trainImgList


def decomposition(img, k):
    """
    将一张图片进行分解,将Ui*ViT存入到List中
    :param img: 传入的一张图片
    :param k: 基是U1*V1T+...+Uk*VkT
    :return:
    """
    U, E, V = np.linalg.svd(img)
    # U, E, V = SVD(img)
    bases = []
    m, n = img.shape
    for i in range(k):
        u = np.array(U.T[i]).reshape(m, 1)
        v = np.array(V[i]).reshape(n, 1)
        bases.append((u, v))
    bases = np.array(bases, dtype=object)
    return E, bases


def getVector(img, base):
    """
    根据 τ_i=u_i^T*B*v_i 求得图像B在图像A的基下的坐标
    :param img:
    :param base:
    :return:
    """
    len = base.shape[0]
    B = img
    T = []
    for i in range(len):
        u_i, v_i = base[i]
        T.append(np.dot(np.dot(u_i.T, B), v_i))
    return np.array(T).reshape(k)


def getDistance(person_img, base):
    vector = getVector(person_img, base[1])
    baseVector = base[0][:k]
    # 计算这个人的在基下的坐标和原图的坐标之间的距离
    return np.linalg.norm(vector - baseVector)


def classify(person_img):
    person_number = 999
    min = 99999999
    # 计算出每个人的最小距离,如果对检测图进行检测,距离大于最小距离,则表示这张图是这个人
    for i in range(40):
        tmp_distance = getDistance(person_img, img_bases[i])
        if min > tmp_distance:
            person_number = i
            min = tmp_distance
    return person_number


img_bases = []  # img_bases存放着40个人的基信息,每个人的基信息中存有原有的坐标和基
k = 10  # 取前k个特征值
if __name__ == '__main__':
    start = time.time()
    imgList = img2array()
    # 以每个人第一张图作为标准,存储不同人的基
    for i in range(40):
        E, base = decomposition(imgList[10 * i], k)
        img_bases.append((E, base))
    right_num = 0  # 最终正确的数量
    test_num = len(imgList)  # 要测试的数量
    for i in range(test_num):
        classified = classify(imgList[i])
        fact = (i - i % 10) / 10
        if classified == fact:
            right_num += 1
        else:
            print(f"第{i + 1}张图像预估属于第{classified + 1}类,但实际属于第{int(fact + 1)}类")
    print("准确率:%.2f%%" % (right_num / float(test_num) * 100))
    print(round(time.time() - start, 3), "秒")
    # person1 = imgList[:10]
    # person1_img = getMeanPerson(person1)
    # cv2.imwrite("tmp/person1.png", person1_img)
    # 写入测试
    # A = decomposition(imgList[0], 10)
    # tmp_img = A[0]
    # for i in range(1, len(A)):
    #     cv2.imwrite("tmp/img" + str(i) + ".png", tmp_img)
    #     tmp_img += A[i]
