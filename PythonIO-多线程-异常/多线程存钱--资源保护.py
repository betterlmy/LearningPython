# author:TYUT-Lmy
# date:2021/12/13
# description:
from time import sleep
from threading import Thread


class Account:
    def __init__(self):
        self._balance = 0

    @property
    def balance(self):
        return self._balance

    def deposit(self, amount):
        """
        存款
        :param amount: 存款的金额
        :return:
        """
        new_balance = self.balance + amount
        sleep(0.1)
        self._balance = new_balance


class AddMoneyThread(Thread):
    def __init__(self, account, amount):
        super().__init__()
        self._account = account
        self._amount = amount

    def run(self):
        self._account.deposit(self._amount)


def main():
    account = Account()
    threads = []
    for _ in range(100):
        t = AddMoneyThread(account, 1)
        threads.append(t)
        t.start() # start之后 会自动执行对象的run()方法

    for t in threads:
        t.join()

    print(f"账户的余额是{account.balance}")


if __name__ == "__main__":
    main()