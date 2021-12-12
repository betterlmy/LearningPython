# author:TYUT-Lmy
# date:2021/12/12
# description:
# 多线程同时下载
from multiprocessing import Process  # 使用该模块创建子线程
from os import getpid
from random import randint
from time import time, sleep


def download_task(filename="123.txt",time_to_download=None):
    """
    下载单个任务,随机下载时间
    :param filename: 下载的文件名
    :return:
    """
    print(f"开始下载{filename},PID={getpid()}")
    if not time_to_download:
        time_to_download = randint(2, 3)
    sleep(time_to_download)
    print(f"{filename}下载完成,耗时{time_to_download}秒")


def main():
    start = time()
    process1 = Process(target=download_task, args=("Python中文版.pdf",10,))
    process1.start()
    process3 = Process(target=download_task, args=("东京热.avi",))
    process3.start()

    process1.join()
    process3.join()
    # 使用join()方法,等待两个线程执行结束,否则会直接输出后续的内容
    # 进程的join顺序没有影响
    end = time()
    print(f"全部下载完成,耗时{round(end - start, 2)}秒")


if __name__ == "__main__":
    main()
