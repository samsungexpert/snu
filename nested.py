

import numpy as np


def outer(a, b):
    def inner(c):
        return a+b+c+1
    return inner



print(outer(1,2)(5))

