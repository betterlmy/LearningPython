# author:TYUT-Lmy
# date:2021/12/14
# description:
from socket import socket, AF_INET, SOCK_STREAM
from json import loads
from base64 import b64decode
import os


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def main():
    client = socket(family=AF_INET, type=SOCK_STREAM)
    port = 7880
    ip_port = ("127.0.0.1", port)
    while True:
        try:
            client.connect(ip_port)
            print("连接成功")
            break
        except ConnectionError:
            print("连接被拒绝,尝试重新连接")
            port += 1
            ip_port = ("127.0.0.1", port)

    # 定义一个保存二进制数据的对象
    in_data = bytes()
    rec_size = 1024
    data = client.recv(rec_size)
    print("接受了数据")
    while data:
        # 将收到的数据拼接起来
        print("接受了数据")
        in_data += data
        data = client.recv(rec_size)
    # 将收到的二进制数据解码成JSON字符串并转换成字典
    # loads函将JSON字符串转成字典对象
    my_dict = loads(in_data.decode('utf-8'))
    filename = my_dict['filename']
    filedata = my_dict['filedata'].encode('utf-8')

    dir = "./res/new/"
    check_dir(dir)
    with open(dir + filename, 'wb') as f:
        # 将base64格式的数据解码成二进制数据并写入文件
        f.write(b64decode(filedata))
    print('图片已保存.')


if __name__ == '__main__':
    main()
