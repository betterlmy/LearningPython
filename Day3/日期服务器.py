# author:TYUT-Lmy
# date:2021/12/14
# description:
from socket import socket, SOCK_STREAM, AF_INET
from datetime import datetime


def main():
    # 1.创建套接字并指定使用哪种服务
    server = socket(family=AF_INET, type=SOCK_STREAM)
    # 2.绑定IP地址和端口(打开该端口,作为服务端)
    ip_port = ('127.0.0.1', 6743)
    server.bind(ip_port)
    # 3.开启监听,参数可以理解为连接队列的大小
    server.listen(2)
    print("服务器开启了监听")
    while True:
        # 4.通过循环接收客户端的连接,并作出相应的处理
        # accept 方法是一个阻塞方法, 如果没有客户端连接到服务器端口,则不会向下执行,返回一个元组,其中第一个元素是客户端对象,第二个元素是客户端的IP和端口
        client, addr = server.accept()
        print(f"{str(addr)}连接了服务器.")

        # 5.检测到有客户端连接,则发送数据
        client.send(f"当前北京时间{str(datetime.now())}\n".encode('utf-8'))

        print(f"向服务器发送了日期信息.")

        # 6.发送完成,断开客户端
        client.close()


if __name__ == "__main__":
    main()
