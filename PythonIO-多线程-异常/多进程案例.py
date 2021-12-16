# author:TYUT-Lmy
# date:2021/12/13
# description:
from multiprocessing import Process, Queue
from time import time


def task_handler(start_num, result_queue):
    total = 0
    for number in range(start_num, start_num + 125000000):
        total += number
    result_queue.put(total)


def main():
    now = time()
    processes = []
    result_queue = Queue()
    for i in range(8):
        p = Process(target=task_handler,
                    args=(1 + 125000000 * i, result_queue))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    total = 0
    while not result_queue.empty():
        total += result_queue.get()

    print(total)
    print(f"程序执行了{round(time() - now, 4)}秒")


if __name__ == '__main__':
    main()
