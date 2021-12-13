# author:TYUT-Lmy
# date:2021/12/13
# description:
from multiprocessing import Process, Queue
from threading import Thread, Lock
from time import time


class Task(Thread):

    def __init__(self, start_num):
        super().__init__()
        self._start_num = start_num
        self._lock = Lock()
        self.SUM = 0

    @property
    def start_num(self):
        return self._start_num

    def run(self):
        for number in range(self.start_num, self.start_num + 25000000):
            self.SUM += number

    def get_sum(self):
        return self.SUM


def task_handler(start_num, result_queue):
    total = 0
    tasks = []
    for i in range(5):
        t = Task(start_num + i * 25000000)
        tasks.append(t)
        t.start()
    for t in tasks:
        t.join()
        total += t.get_sum()

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
    # 500000000500000000
    # 程序执行了12.6846秒


if __name__ == '__main__':
    main()
