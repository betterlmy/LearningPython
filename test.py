# author:TYUT-Lmy
# date:2021/12/8
# description:

def foo():
    print(1)
    pass


def bar():
    pass

def test(num1):
    def add(num2):
        return num1 + num2
    return add


print(2)
# __name__是Python中一个隐含的变量它代表了模块的名字
# 只有被Python解释器直接执行的模块的名字才是__main__
if __name__ == '__main__':
    print('call foo()')
    foo()
    print('call bar()')
    bar()
    a = test(1)
    b = test(2)
    print(a(1))
    print(b(1))
