import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')

plt.rcParams["font.sans-serif"] = ["PingFang HK"]
plt.rcParams["axes.unicode_minus"] = False


# 目标函数
def default(x):
    y = np.sum(x * x)
    return y


class FPA:
    def __init__(self, num_pop, dim, ub, lb, f_obj, verbose):
        self.num_pop = num_pop  # 种群数目
        self.dim = dim  # 维数
        self.ub = ub  # 上界
        self.lb = lb  # 下界
        self.pop = np.empty((num_pop, dim))  # 种群
        self.f_obj = f_obj  # 目标函数
        self.f_score = np.empty((num_pop, 1))
        self.verbose = verbose  # 显示

        self.p_best = None  # 最好个体
        self.f_best = None  # 最好适应度值

        self.iter_f_score = []

    def initialize(self):
        """
        种群初始化
        :return: 初始化种群和种群分数
        """
        num_boundary = len(self.ub)
        if num_boundary == 1:
            for i in range(self.num_pop):
                self.pop[i, :] = self.lb + (self.ub - self.lb) * np.random.rand(1, self.dim).flatten()
                self.f_score[i] = self.f_obj(self.pop[i, :])
        else:
            for i in range(self.dim):
                self.pop[:, i] = self.lb[i] + (self.ub[i] - self.lb[i]) * np.random.rand(self.num_pop, 1).flatten()
            for i in range(self.num_pop):
                self.f_score[i] = self.f_obj(self.pop[i, :])
        return [self.pop, self.f_score]

    def get_best(self):
        """
        获取最优解
        :return: 最优索引和最优得分
        """
        idx = np.argmin(self.f_score)
        self.p_best = self.pop[idx, :]
        self.f_best = np.min(self.f_score)
        return [self.p_best, self.f_best]

    def Levy(self, beta=1.5):
        """
        莱维飞行
        :param beta: 固定参数
        :return: 随机数
        """
        sigma = (np.random.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                np.random.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        # print(sigma)
        u = np.random.normal(0, sigma, (1, self.dim))
        v = np.random.normal(0, 1, (1, self.dim))
        levy = u / (np.abs(v) ** (1 / beta))
        return levy

    def clip(self, p):
        """
        :param p: 花粉配子
        :return: 越界处理后的花粉配子
        """
        i = p < self.lb
        p[i] = np.array(self.lb)[i]
        j = p > self.ub
        p[j] = np.array(self.ub)[j]
        return p

    def fit(self, max_generation, p=0.8):
        """
        算法迭代
        :param max_generation: 最大迭代次数
        :param p: 切换概率
        :return: 最优值和最优个体
        """
        self.pop, self.f_score = self.initialize()  # 初始化
        self.p_best, self.f_best = self.get_best()  # 当前最优个体

        self.iter_f_score.append(self.f_best)

        # 迭代
        for i in range(max_generation):
            for j in range(self.num_pop):
                if np.random.rand() < p:
                    # 异花授粉公式
                    new_pop = self.pop[j, :] + self.Levy() * (self.pop[j, :] - self.p_best)
                else:
                    # 自花授粉公式
                    idx_set = np.random.choice(range(self.num_pop), 2)
                    new_pop = self.pop[j, :] + np.random.rand(1, self.dim) * (self.pop[idx_set[0], :] -
                                                                              self.pop[idx_set[1], :])

                # 越界处理
                new_pop = self.clip(new_pop.flatten())
                new_f_score = self.f_obj(new_pop)

                # 进化算法
                if new_f_score < self.f_score[j]:
                    self.pop[j, :] = new_pop
                    self.f_score[j] = new_f_score

                if new_f_score < self.f_best:
                    self.p_best = new_pop
                    self.f_best = new_f_score

            self.iter_f_score.append(self.f_best)

            if self.verbose:
                print("============{}/{}==============".format(i, max_generation))
                print(self.f_best)

        return [self.iter_f_score, self.f_best]


# 测试
if __name__ == "__main__":
    n_pop = 50
    d = 30
    upper = [100, ]*d
    lower = [-100,]*d

    max_iter = 3000

    fpa = FPA(n_pop, d, upper, lower, default, True)
    iter_score, _ = fpa.fit(max_iter)

    xx = [int(i) for i in range(0, max_iter+1, 50)]
    yy = np.array(iter_score)
    print(fpa.p_best)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.title("收敛曲线")
    plt.plot(xx, yy[xx], c='k', marker='.')
    plt.xlim([0, max_iter])
    plt.xlabel("迭代次数")
    plt.ylabel("适应度值")
    plt.show()
