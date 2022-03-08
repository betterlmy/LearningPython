#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

train = np.load("./train.npy")
label = np.load("./train_label.npy")

def svd_decompose(img):
    u, lamda, v = np.linalg.svd(img)
    return u, lamda, v

def restore1(u, sigma, v, k):
    m = len(u)
    n = len(v)
    print(m, n)
    a = np.zeros((m, n))
    # 重构图像
    # a = np.dot(u[:, :k], np.diag(sigma[:k])).dot(v[:k, :])
    # 上述语句等价于：
    for i in range(k):
        ui = u[:, i].reshape(m, 1)
        vi = v[i].reshape(1, n)
        a += sigma[i] * np.dot(ui, vi)
    a[a < 0] = 0
    a[a > 255] = 255
    return np.rint(a).astype('uint8')

img = train[0]
print(img.shape)
print(img)

u_b, lamda_b, v_b = svd_decompose(img[:, :, 0])
u_g, lamda_g, v_g = svd_decompose(img[:, :, 1])
u_r, lamda_r, v_r = svd_decompose(img[:, :, 2])
print(u_r.shape, lamda_r.shape, v_r.shape)

plt.figure(facecolor = 'w', figsize = (10, 10))
# 实验表示奇异值选择五十个或者六十个是ok的
# 奇异值个数依次取：1,2,...,12
K = 224
i = 0 
for k in range(20, 30):
    i += 1
    R = restore1(u_r, lamda_r, v_r, k)
    G = restore1(u_g, lamda_g, v_g, k)
    B = restore1(u_b, lamda_b, v_b, k)
    I = np.stack((R, G, B), axis = 2)   # plt的通道顺序为BGR
    # I = np.stack((B, G, R), axis = 2)

    # 将图片重构后的显示出来
    plt.subplot(3, 4, i)
    plt.imshow(I)
    plt.axis('off')
    plt.title(u'number of eigenvalue: %d' %  k)

plt.suptitle(u'SVD', fontsize = 20)
plt.tight_layout(0.2, rect = (0, 0, 1, 0.92))
plt.show()