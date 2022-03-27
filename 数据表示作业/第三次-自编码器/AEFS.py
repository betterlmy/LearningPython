from icecream import ic
import torch, torchvision
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from torch.utils import data

import lmy


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 12),
            nn.Sigmoid(),
            nn.Linear(12, 3),  # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def loadMnistData(batch_size, root="./data", size=None):
    """下载FashionMnist数据集并加载到内存中

    :param root:
    :param batch_size:
    :param size:需要进行resize 修改长宽
    :return:返回训练集和测试集的DataLoader
    """
    # 通过ToTenser()这个类 将图像数据从PIL类型转为浮点型的tensor类型,并除以255使得所有的像素数值均在0-1之间(归一化)
    trans = [transforms.ToTensor()]
    if size:
        trans.insert(0, transforms.Resize(size))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root=root, train=True, transform=trans, download=False)
    # mnist_test = torchvision.datasets.MNIST(root=root, train=False, transform=trans, download=False)
    print("数据集加载成功", len(mnist_train))  # 60000 ,10000
    num_workers = 4  # 设置读取图片的进程数量 小于cpu的核心数

    # 展示图像
    # plt.imshow(mnist_train.data[0].numpy(), cmap='gray')
    # plt.title('%i' % mnist_train.targets[0])
    # plt.show()
    return mnist_train, data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)

    # 初始化自定义网络


def squared_loss(y_hat, y):
    """
    计算损失值
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def train_epoch(net, train_iter, loss, updater, batch_size):
    if isinstance(net, torch.nn.Module):
        net.train()  # 设置为训练模式
    for X, _ in train_iter:
        X = X.reshape(batch_size, 784)
        encoded, decoded = net.forward(X)
        l = loss(decoded, X)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater.step(net)


if __name__ == '__main__':
    # 加载数据集
    batch_size = 10
    mnist_train, train_iter = loadMnistData(batch_size)

    # 设定参数
    input_size = 784
    hidden_size = 32
    output_size = 784

    # 初始化网络
    encoderDecoder = EncoderDecoder()

    loss = squared_loss
    batch_size = 10
    num_epochs = 10
    lr = .1
    # updater = lmy.SGD(encoderDecoder, lr=lr, batch_size=10)
    updater = torch.optim.Adam(encoderDecoder.parameters(), lr=lr)
    for epoch in range(num_epochs):
        print(f"epoch{epoch}")
        train_epoch(encoderDecoder, train_iter, loss, updater, batch_size)
    print(f"训练完成")
    T1 = mnist_train.data[0].type(torch.float32).detach().reshape(1, -1)
    _, decoded = encoderDecoder.forward(T1)
    decoded = decoded.reshape(28, 28).detach()
    plt.imshow(decoded.numpy(), cmap='gray')
    plt.show()
