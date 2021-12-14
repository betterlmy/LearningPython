# author:TYUT-Lmy
# date:2021/12/14
# description:
from socket import socket, AF_INET, SOCK_STREAM
from time import sleep
from base64 import b64encode
from json import dumps
from threading import Thread


def get_data() -> str:
    """
    guido.jpg是一个图片,即二进制数据,不能够通过json传输,需要将二进制编码
    :return: 图片的二进制数据
    """
    with open("./res/guido.jpg", 'rb') as f:
        DATA = b64encode(f.read()).decode('utf-8')
    return DATA


class FileTransferHandler(Thread):
    def __init__(self, client, addr, data):
        super().__init__()
        self.client = client
        self.addr = addr
        self.data = data
        print(f"针对{addr}的线程已经初始化完成")

    def run(self):
        # 生成包含文件名和内容的字典

        self.client.send(self.data)
        self.client.close()
        print(f"{self.addr}已经关闭")


def main():
    data = get_data()  # 获取图片的数据

    # 配置服务器
    server = socket(family=AF_INET, type=SOCK_STREAM)
    port = 7880
    ip_port = ("127.0.0.1", port)
    while True:
        try:
            server.bind(ip_port)
            break
        except OSError:
            print(f"端口{port}已经被占用")
            port += 1
            ip_port = ("127.0.0.1", port)

    server.listen()  # 指定最大连接的数量
    print(f'服务器启动开始监听{port}端口')

    # 设置监听动作
    while True:
        sleep(.1)
        client, addr = server.accept()
        print(f"已经连接{addr}")
        # 不断启动线程来对客户端的请求进行处理
        my_dict = {'file_name': 'guido.jpg', 'file_data': data}
        # 将字典通过dumps转为json 用于传输
        json_str = dumps(my_dict)
        FileTransferHandler(client, addr, json_str).start()


if __name__ == '__main__':
    main()
