# author:TYUT-Lmy
# date:2021/12/16
# description:
"""
迭代工具模块
"""
import itertools

# 产生ABCD的全排列
a = itertools.permutations('ABCD')  # 产生一个可迭代对象
# for i in a:
#     print(i)
# ABCDE中产生五选三的组合
b = itertools.combinations('ABCDE', 3)
# for i in b:
#     print(i)
# 产生ABCD和123的笛卡尔积
c = itertools.product('ABCD', '123')
# for i in c:
#     print(i)
# 产生ABC的无限循环序列
# d = itertools.cycle(('A', 'B', 'C'))
# for i in d:
#     print(i)
pass