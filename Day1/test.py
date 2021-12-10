# author:TYUT-Lmy
# date:2021/12/9
# description:
from random import choice

import numpy as np

def main(*args):
    a = [1, 2, 3]
    b = ["1", "2"]
    attacked = choice(b)
    print(attacked)
    b.remove(attacked)
    print(b)


if __name__ == '__main__':
    main()
