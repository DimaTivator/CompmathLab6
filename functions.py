import numpy as np


def f1(x, y):
    return 4 * x + y / 3


def fy1(x, c):
    return c * np.exp(x / 3) - 12 * x - 36


def c1(x, y):
    return (y + 12 * x + 36) / np.exp(x / 3)


def f2(x, y):
    return x ** 2 + y


def fy2(x, c):
    return c * np.exp(x) - x ** 2 - 2 * x - 2


def c2(x, y):
    return (-y - x ** 2 - 2 * x - 2) / (-np.exp(x))


def f3(x, y):
    return y * np.cos(x)


def fy3(x, c):
    return c * np.exp(np.sin(x))


def c3(x, y):
    return y / np.exp(np.sin(x))
