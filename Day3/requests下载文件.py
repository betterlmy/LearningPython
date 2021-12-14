# author:TYUT-Lmy
# date:2021/12/14
# description:
from threading import Thread
import requests
import os


class DownloadHandler(Thread):
    def __init__(self, url, path):
        super().__init__()
        self.url = url
        self.path = path

    def run(self):
        filename = self.url[self.url.rfind("/") + 1:]  # 将url最后一个/后面所有的文件设置为文件名
        response = requests.get(self.url)

        with open(self.path + "/" + filename, 'wb') as f:
            f.write(response.content)


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    # 通过requests模块的get函数获取网络资源
    # 下面的代码中使用了天行数据接口提供的网络API
    # 要使用该数据接口需要在天行数据的网站上注册
    # 然后用自己的Key替换掉下面代码的中APIKey即可
    resp = requests.get('https://api.tianapi.com/esports/index?key=acf20251be985280976b251ac3dbbc91&num=10')
    data_model = resp.json()  # 返回json数据,需要解析为字典
    path = "./res"
    check_path(path)
    for mm_dict in data_model["newslist"]:
        url = mm_dict['picUrl']
        DownloadHandler("http:" + url, path).start()
    print(f"下载{len(data_model['newslist'])}张图片完成")


if __name__ == '__main__':
    main()