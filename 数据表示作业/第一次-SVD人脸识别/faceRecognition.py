import time

import cv2
import numpy as np

# from svd import SVD


def img2array():
    # 将所有图片读取为ndarrayList
    path = "./att_faces/s"
    imgList = []

    for _ in range(1, 41):
        for __ in range(1, 11):
            imgPath = path + str(_) + "/" + str(__) + ".pgm"
            img = cv2.imread(imgPath)  # 通过opencv读取图片
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转为灰度图
            imgList.append(img)

    imgList = np.array(imgList)
    return imgList


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
    return bases


def getVector(img, base_num):
    """
    根据 τ_i=u_i^T*B*v_i 求得图像B在图像A的基下的坐标
    :param img:要获取坐标的图像
    :param base_num:第几个人的基
    :return:
    """
    base = img_bases[base_num]
    len = base.shape[0]
    B = img
    T = []
    for i in range(len):
        u_i, v_i = base[i]
        T.append(np.dot(np.dot(u_i.T, B), v_i))
    return np.array(T).reshape(k)


def getDistance(person_img, person):
    vector = getVector(person_img, person)
    baseVector = ave_vectors[person]
    # 计算这个人的在基下的坐标和原图的坐标之间的距离
    return np.linalg.norm(vector - baseVector)


def classify(person_img):
    person_number = 999
    min = 99999999
    # 计算出每个人的最小距离,如果对检测图进行检测,距离大于最小距离,则表示这张图是这个人
    for i in range(40):
        # 获取这个人的照片和40个基的距离
        tmp_distance = getDistance(person_img, i)
        if min > tmp_distance:
            person_number = i
            min = tmp_distance
    return person_number


def getAveVector(person_num):
    # 训练函数
    # 获取这个人的平均向量
    vector = np.zeros([k])
    for i in range(person_num * 10, person_num * 10 + train_num):
        vector += getVector(imgList[i], int((person_num - person_num % 10) / 10))
    vector /= train_num
    return vector


def combine():
    for person_num in range(40):
        vector = ave_vectors[person_num]
        base = img_bases[person_num]
        m = base[0][0].shape[0]
        n = base[0][1].shape[0]
        base_img = np.zeros([m, n])
        for i in range(k):
            base_img += vector[i] * np.dot(base[i][0], base[i][1].T)
        cv2.imwrite("tmp/person" + str(person_num) + ".png", base_img)


img_bases = []  # img_bases存放着40个人的基信息,每个人的基信息中存有原有的坐标和基
k = 10  # 取前k个特征值
train_num = 8  # 每个人的8个用来训练
test_num = 10 - train_num
ave_vectors = []

if __name__ == '__main__':
    start = time.time()
    imgList = img2array()
    # 添加训练 获取平均的距离
    for i in range(40):
        # 以每个人第一张图作为标准,存储不同人的基
        base = decomposition(imgList[10 * i], k)
        # base[0]
        img_bases.append(base)
    for i in range(40):
        # 在每个人的基上,添加每个人8张照片的平均向量
        ave_vectors.append(getAveVector(i))
    combine()
    right_num = 0  # 最终正确的数量
    # 将所有的需要测试的图片放到单独的图片List中进行测试
    testImgList = []
    for i in range(40):
        for index in range(i * 10 + train_num, (i + 1) * 10):
            testImgList.append((index, imgList[index]))
    for testImg in testImgList:
        classified = classify(testImg[1])  # 预测的值
        fact = int((testImg[0] - testImg[0] % 10) / 10)  # 实际的值
        if classified == fact:
            right_num += 1
        else:
            print(f"第{testImg[0] + 1}张图像预估属于第{classified + 1}类,但实际属于第{int(fact + 1)}类")
    print("准确率:%.2f%%" % (right_num / float(test_num) / 40 * 100))

    # person1 = imgList[:10]
    # person1_img = getMeanPerson(person1)
    # cv2.imwrite("tmp/person1.png", person1_img)
    # 写入测试
    # A = decomposition(imgList[0], 10)
    # tmp_img = A[0]
    # for i in range(1, len(A)):
    #     cv2.imwrite("tmp/img" + str(i) + ".png", tmp_img)
    #     tmp_img += A[i]
    print(round(time.time() - start, 3), "秒")
