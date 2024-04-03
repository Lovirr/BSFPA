import numpy as np


# 单峰测试函数
# Sphere Function
def fun1(x):
    y = np.sum(x * x)
    return y


problem1 = {
    "dim": 30,
    "lb": [-100, ] * 30,
    "ub": [100, ] * 30,
    "best": 0
}


# Schwefel's Problem 2.22
def fun2(x):
    y = np.sum(np.abs(x)) + np.prod(np.abs(x))
    return y

problem2 = {
    "dim": 30,
    "lb": [-10, ] * 30,
    "ub": [10, ] * 30,
    "best": 0
}


# Schwefel's Problem 1.2
def fun3(x):
    y = 0
    for i in range(len(x)):
        y = y + np.square(np.sum(x[0:i + 1]))
    return y

problem3 = {
    "dim": 30,
    "lb": [-100, ] * 30,
    "ub": [100, ] * 30,
    "best": 0
}


# Schwefel's Problem 2.21
def fun4(x):
    y = np.max(np.abs(x))
    return y

problem4 = {
    "dim": 30,
    "lb": [-100, ] * 30,
    "ub": [100, ] * 30,
    "best": 0
}

# Generalized Rosenbrock's Function
def fun5(x):
    x_len = len(x)
    y = np.sum(100 * np.square(x[1:x_len] - np.square(x[0:x_len - 1]))) + np.sum(np.square(x[0:x_len - 1] - 1))
    return y

problem5 = {
    "dim": 30,
    "lb": [-30, ] * 30,
    "ub": [30, ] * 30,
    "best": 0
}

# Step Function
def fun6(x):
    y = np.sum(np.square(np.abs(x + 0.5)))
    return y

problem6 = {
    "dim": 30,
    "lb": [-100, ] * 30,
    "ub": [100, ] * 30,
    "best": 0
}

# Quartic Function i.e. Noise
def fun7(x):
    i = np.arange(1, len(x) + 1)
    y = np.sum(i * (x ** 4)) + np.random.random()
    return y

problem7 = {
    "dim": 30,
    "lb": [-1.28, ] * 30,
    "ub": [1.28, ] * 30,
    "best": 0
}

# 多峰测试函数
# Generalized Schwefel's Problem 2.26
def fun8(x):
    y = np.sum(-x * np.sin(np.sqrt(np.abs(x))))
    return y

problem8 = {
    "dim": 30,
    "lb": [200, ] * 30,
    "ub": [500, ] * 30,
    "best": -12569.5
}

# Generalized Rastrigin's Function
def fun9(x):
    dim = len(x)
    y = np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim
    return y

problem9 = {
    "dim": 30,
    "lb": [-5.12, ] * 30,
    "ub": [5.12, ] * 30,
    "best": 0
}

# Ackley's Function
def fun10(x):
    dim = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    sum_1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / dim))
    sum_2 = np.exp(np.sum(np.cos(c * x)) / dim)
    y = sum_1 - sum_2 + a + np.exp(1)
    return y

problem10 = {
    "dim": 30,
    "lb": [-32, ] * 30,
    "ub": [32, ] * 30,
    "best": 0
}

# Generalized Griewank's Function
def fun11(x):
    dim = len(x)
    i = np.arange(1, dim + 1)
    y = np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1
    return y

problem11 = {
    "dim": 30,
    "lb": [-600, ] * 30,
    "ub": [600, ] * 30,
    "best": 0
}

# Generalized Penalized Function 1
def Ufun(x, a, k, m):
    dim = len(x)
    U = np.zeros(dim)
    for i in range(len(x)):
        if x[i] > a:
            U[i] = k * ((x[i] - a) ** m)
        elif x[i] < -a:
            U[i] = k * ((-x[i] - a) ** m)
        else:
            U[i] = 0
    return U


def fun12(x):
    dim = len(x)
    pi = np.pi
    sum_1 = (np.pi / dim) * (10 * ((np.sin(pi * (1 + (x[0] + 1) / 4))) ** 2)
                             + np.sum((((x[:dim - 2] + 1) / 4) ** 2) *
                                      (1 + 10 * ((np.sin(pi * (1 + (x[1:dim - 1] + 1) / 4)))) ** 2))
                             + ((x[dim - 1]) / 4) ** 2)
    sum_2 = np.sum(Ufun(x, 10, 100, 4))
    y = sum_1 + sum_2
    return y

problem12 = {
    "dim": 30,
    "lb": [-50 ] * 30,
    "ub": [50, ] * 30,
    "best": 0
}

# Generalized Penalized Function 2
def fun13(x):
    dim = len(x)
    pi = np.pi
    y = 0.1 * ((np.sin(3 * pi * x[0])) ** 2 + np.sum(
        ((x[0:dim - 2]) - 1) ** 2 * (1 + (np.sin(3 * pi * x[1:dim - 1])) ** 2)))
    +((x[dim - 1] - 1) ** 2) * (1 + (np.sin(2 * pi * x[dim - 1])) ** 2) + np.sum(Ufun(x, 5, 100, 4))
    return y

problem13 = {
    "dim": 30,
    "lb": [-50 ] * 30,
    "ub": [50, ] * 30,
    "best": 0
}

# 固定多峰测试函数
# Shekel's Foxholes Function
def fun14(x):
    aS = np.array(
        [[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
         [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    bS = np.zeros(25)
    for j in range(0, 25):
        bS[j] = np.sum((x.T - aS[:, j]) ** 6)
    y = (1 / 500 + np.sum(1 / (np.arange(1, 25 + 1) + bS))) ** (-1)
    return y

problem14 = {
    "dim": 2,
    "lb": [-65.536, ] * 2,
    "ub": [65.536, ] * 2,
    "best": 0.99800383
}

# Kowalik's Function
def fun15(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    bK = 1 / bK
    y = np.sum((aK - ((x[0] * (bK ** 2 + x[1] * bK)) / (bK ** 2 + x[2] * bK + x[3]))) ** 2)
    return y

problem15 = {
    "dim": 4,
    "lb": [-5, ] * 4,
    "ub": [5, ] * 4,
    "best": 0.0003075
}

# Six-Hump Camel-Back Function
def fun16(x):
    y = 4 * (x[0] ** 2) - 2.1 * (x[0] ** 4) + (x[0] ** 6) / 3 + x[0] * x[1] - 4 * (x[1] ** 2) + 4 * (x[1] ** 4)
    return y

problem16 = {
    "dim": 2,
    "lb": [-5, ] * 2,
    "ub": [5, ] * 2,
    "best": 1.03162845
}

# Branin Function
def fun17(x):
    pi = np.pi
    y = ((x[1]) - (x[0] ** 2) * 5.1 / (4 * (pi ** 2)) + 5 / pi * x[0] - 6) ** 2 + 10 * (1 - 1 / (8 * pi)) * np.cos(
        x[0]) + 10
    return y

problem17 = {
    "dim": 2,
    "lb": [-5, 10],
    "ub": [0, 15] ,
    "best": 0.39788735
}

# Goldstein-Price Function
def fun18(x):
    y = (1 + ((x[0] + x[1] + 1) ** 2) * (
            19 - 14 * x[0] + 3 * (x[0] ** 2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (x[1] ** 2))) * (
                30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                18 - 32 * x[0] + 12 * (x[0] ** 2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1] ** 2)))
    return y

problem18 = {
    "dim": 2,
    "lb": [-2, ] * 2,
    "ub": [2, ] * 2 ,
    "best": 2.9999999
}

# Hartman's Family
def fun19(X):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747],
                   [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    y = 0
    for i in range(0, 4):
        y = y - cH[i] * np.exp(-(np.sum(aH[i] * ((X - pH[i]) ** 2))))
    return y

problem19 = {
    "dim": 3,
    "lb": [0, ] * 3,
    "ub": [1, ] * 3,
    "best": -3.86278214
}

def fun20(x):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array(
        [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.413, 0.8307, 0.3736, 0.1004, 0.9991],
         [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    y = 0
    for i in range(0, 4):
        y = y - cH[i] * np.exp(-(np.sum(aH[i] * ((x - pH[i]) ** 2))))
    return y

problem20 = {
    "dim": 6,
    "lb": [0, ] * 6,
    "ub": [1, ] * 6,
    "best": -3.32199517
}

# Shekel's Family
def fun21(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([[0.1], [0.2], [0.2], [0.4], [0.4],
                    [0.6], [0.3], [0.7], [0.5], [0.5]])
    y = 0
    for i in range(0, 5):
        y = y - (np.sum((x - aSH[i]) ** 2) + cSH[i]) ** (-1)
    return y

problem21 = {
    "dim": 4,
    "lb": [0, ] * 4,
    "ub": [10, ] * 4,
    "best": -10.1531996
}

def fun22(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([[0.1], [0.2], [0.2], [0.4], [0.4],
                    [0.6], [0.3], [0.7], [0.5], [0.5]])
    y = 0
    for i in range(0, 7):
        y = y - (np.sum((x - aSH[i]) ** 2) + cSH[i]) ** (-1)
    return y

problem22 = {
    "dim": 4,
    "lb": [0, ] * 4,
    "ub": [10, ] * 4,
    "best": -10.4029405
}

def fun23(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([[0.1], [0.2], [0.2], [0.4], [0.4],
                    [0.6], [0.3], [0.7], [0.5], [0.5]])
    y = 0
    for i in range(0, 10):
        y = y - (np.sum((x - aSH[i]) ** 2) + cSH[i]) ** (-1)
    return y

problem23 = {
    "dim": 4,
    "lb": [0, ] * 4,
    "ub": [10, ] * 4,
    "best": -10.5364
}