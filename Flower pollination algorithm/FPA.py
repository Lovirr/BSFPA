import numpy as np
from matplotlib import pyplot as plt
import random

# 初始化种群
def init(n_pop, lb, ub, nd):
    """
    :param n_pop: 种群
    :param lb: 下界
    :param ub: 上界
    :param nd: 维数
    """
    p = lb + (ub - lb) * np.random.rand(n_pop, nd)
    return p


# 适应度函数

  # 函数句柄


# Levy飞行Beale
def Levy(nd, beta=1.5):
    num = np.random.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = np.random.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)

    u = np.random.normal(0, sigma_u ** 2, (1, nd))
    v = np.random.normal(0, 1, (1, nd))

    z = u / (np.abs(v) ** (1 / beta))
    return z


def FPA(Max_g, n_pop, Pop, nd, lb, ub, detail):  # FPA算法
    """
    :param Max_g: 迭代次数
    :param n_pop: 种群数目
    :param Pop: 花粉配子
    :param nd: 维数
    :param lb: 下界
    :param ub: 上界
    :param detail: 显示详细信息
    """
    # 计算初始种群中最好个体适应度值
    pop_score = f_score(Pop)
    g_best = np.min(pop_score)
    g_best_loc = np.argmin(pop_score)
    g_best_p = Pop[g_best_loc, :].copy()

    # 问题设置
    p = 0.8
    best_fit = np.empty((Max_g,))
    # 迭代
    for it in range(1, Max_g + 1):
        for i in range(n_pop):
            if np.random.rand() < p:
                new_pop = Pop[i, :] + Levy(nd) * (g_best_p - Pop[i, :])
                new_pop = np.clip(new_pop, lb, ub)  # 越界处理
            else:
                idx = random.sample(list(range(n_pop)), 2)
                new_pop = Pop[i, :] + np.random.rand() * (Pop[idx[1], :] - Pop[idx[0], :])
                new_pop = np.clip(new_pop, lb, ub)  # 越界处理
            if f_score(new_pop.reshape((1, -1))) < f_score(Pop[i, :].reshape((1, -1))):
                Pop[i, :] = new_pop
                # 计算更新后种群中最好个体适应度值
        pop_score = f_score(Pop)
        new_g_best = np.min(pop_score)
        new_g_best_loc = np.argmin(pop_score)

        if new_g_best < g_best:
            g_best = new_g_best
            g_best_p = Pop[new_g_best_loc, :].copy()
        best_fit[it - 1] = g_best

        if detail:
            print("----------------{}/{}--------------".format(it, Max_g))
            print(g_best)
            print(g_best_p)

    return best_fit, g_best


if __name__ == "__main__":
    dim = 30  # 变量维度
    pop = init(500, -100, 100, dim)
    fitness, g_best = FPA(1000, 500, pop, dim, -100, 100, False)

    # 可视化
    plt.figure()
    # plt.plot(fitness)
    plt.semilogy(fitness)
    # 可视化
    # fig = plt.figure()
    # plt.plot(p1, fit)
    plt.show()