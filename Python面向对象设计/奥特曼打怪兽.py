# author:TYUT-Lmy
# date:2021/12/10
# description:
from abc import ABCMeta, abstractmethod
from random import randint, random, choice
import numpy as np


class Fighter(object, metaclass=ABCMeta):
    """战斗者"""

    __slots__ = ('_name', '_hp')

    def __init__(self, name, hp):
        """
        初始化方法
        :param name:名字
        :param hp: 血量
        """
        self._name = name
        self._hp = hp

    @property
    def name(self):
        return self._name

    @property
    def hp(self):
        return self._hp

    @staticmethod
    def is_alive(self):
        """
        静态方法，判断是否存活
        :param self:
        :return:
        """
        if self.hp <= 0:
            return False
        return True

    @hp.setter
    def hp(self, hp):
        self._hp = hp

    @abstractmethod
    def attack(self, enemy):
        """
        攻击操作
        :param enemy:攻击的对象
        :return:
        """
        pass


class Ultraman(Fighter):
    __slots__ = ('_name', '_hp', '_mp', '_max_mp')

    def __init__(self, name, hp, mp=100, max_mp=100):
        super().__init__(name, hp)
        self._mp = mp
        self._max_mp = max_mp

    @property
    def mp(self):
        return self._mp

    @property
    def max_mp(self):
        return self._max_mp

    @mp.setter
    def mp(self, mp):
        self._mp = mp

    def utm_attack(self, enemy):
        skill = randint(0, 10)
        if skill <= 9:
            damage = self.attack(enemy)
            print(f"{self.name}对{enemy.name}进行了普通攻击造成了{damage}点伤害,{enemy.name}还剩{enemy.hp}点血")
            enemy.hp -= damage

        else:
            damage = self.huge_attack(enemy, damage1=50)
            if damage is None:
                print(f"{self.name}蓝量不足,无法进行暴击")
            else:
                print(f"{self.name}打出了暴击,对{enemy.name}造成了{damage}点输出")
                enemy.hp -= damage
        if enemy.hp <= 0:
            enemy.hp = 0
            print(f"{enemy.name}已经阵亡")
            return True
        return False

    def attack(self, enemy):
        """
        普通攻击 攻击一个人，无消耗
        :param enemy: 攻击的对象
        :return:
        """
        damage = randint(15, 25)  # 攻击随机掉血
        return damage

    def huge_attack(self, enemy, damage1):
        """
        单体大招攻击
        :param damage1:
        :param enemy: 攻击对象
        :return: 返回是否攻击成功
        """
        cost = 40
        if self.mp > 50:
            damage2 = int(enemy.hp * 3 / 4)
            damage = damage1 if damage1 > damage2 else damage2
            self._mp -= cost
            return damage
        return None

    def resume(self, limit):
        """
        每个回合自动恢复魔法值,有自己的上限
        :return:是否恢复成功
        """
        if self.mp < self.max_mp:
            self.mp += randint(limit[0], limit[1])
            if self.mp > self.max_mp:
                self.mp = self.max_mp
            return True
        return False


def __str__(self):
    """
    获取当前奥特曼的状态
    :return:
    """
    status = f"{self.name}奥特曼的状态:\n血量{self.hp}\n蓝量{self.mp}"
    return status


class Monster(Fighter):
    """怪兽,继承于战斗者，怪兽只会攻击，不能放技能，且没有蓝量"""

    __slots__ = ('_name', '_hp')

    def __init__(self, name, hp):
        super().__init__(name, hp)

    def attack(self, enemy):
        """
        小怪兽只能一个一个的攻击
        :param enemy: 攻击的对象
        :return: 返回是否攻击死亡
        """
        print(f"怪兽{self.name}攻击了{enemy.name},掉了20血,{enemy.name}", end="")
        damage = 22
        enemy.hp -= damage
        if enemy.hp <= 0:
            enemy.hp = 0
            print("已经死亡!")
            return True
        print(f"还有{enemy.hp}血。")
        return False

    def __str__(self):
        """
        获取当前怪兽的状态
        :return:
        """
        status = f"{self.name}怪兽的状态:血量{self.hp}"
        return status


def check_can_fight(*args):
    """
    判断场上是否奥特曼全部阵亡或者怪兽全部阵亡
    :param args: 传入的两个list
    :return: 只要双方有任何一方全部阵亡，则游戏结束 True表明双方都有生存者
    """
    # num = 0
    # for arg in args:
    #     for i in arg:
    #         if i == 1:
    #             num += 1
    #             break
    # if num == 2:
    #     return True
    # return False
    for arg in args:
        if len(arg) == 0:
            return False
    return True


def get_series(num):
    if int(num) > 0:
        return np.random.randint(0, num, 1)[0]
    return None


def fight(ultramans, monsters):
    """
    战斗的经过
    :param ultramans:
    :param monsters:
    :return:
    """

    # 战斗是随机的，谁先打谁后打都随机获得
    all_fighters = ultramans + monsters
    print(f"当前{len(ultramans)}个奥特曼和{len(monsters)}个怪兽正在进行战斗")
    attacker = choice(all_fighters)
    if attacker in ultramans:
        # 先攻击者是奥特曼，随机获取一个怪兽攻击，
        attacked = choice(monsters)
        # 攻击
        if attacker.utm_attack(attacked):
            # 如果攻击死亡 直接移除
            monsters.remove(attacked)

        # 后从怪兽中选择一个攻击奥特曼(如果还有活着的怪兽)
        if len(monsters) > 0:
            new_attacker = choice(monsters)
            new_attacked = choice(ultramans)
            if new_attacker.attack(new_attacked):
                ultramans.remove(new_attacked)

    else:
        # 先攻击者是怪兽，随机获取一个奥特曼攻击
        attacked = choice(ultramans)
        # 攻击
        if attacker.attack(attacked):
            ultramans.remove(attacked)
        if len(ultramans) > 0:
            new_attacker = choice(ultramans)
            new_attacked = choice(monsters)
            if new_attacker.utm_attack(new_attacked):
                monsters.remove(new_attacked)
        for ultraman in ultramans:
            # 所有的还活着的奥特曼需要回蓝
            ultraman.resume([10, 20])

    return ultramans, monsters


def main():
    # 初始化奥特曼和怪兽集
    u1 = Ultraman("艾迪", 500)
    u2 = Ultraman("迪迦", 500, mp=80, max_mp=300)
    u3 = Ultraman("李梦洋", 500, mp=300, max_mp=500)

    m1 = Monster("李飞翔", 500)
    m2 = Monster("王思慧1", 800)
    m3 = Monster("王思慧2", 700)
    m4 = Monster("王思慧3", 700)

    ultramans = [u1, u2, u3]
    monsters = [m1, m2, m3, m4]
    round = 0  # 第一回合

    # 开始战斗
    while check_can_fight(ultramans, monsters):
        # 都活着
        round += 1
        print(f"当前已经战斗到第{round}回合".center(50, '*'))
        ultramans, monsters = fight(ultramans, monsters)

    print("战斗结束", end="")
    if len(ultramans) > 0:
        print("奥特曼胜利")
        return True
    else:
        print("怪兽胜利")
        return False


if __name__ == '__main__':
    num = 0
    rounds = 1000
    for _ in range(rounds):
        if main():
            num += 1
    rate = num * 100 / rounds
    print(f"{rounds}个回合中，奥特曼的胜率只有{rate}%")
