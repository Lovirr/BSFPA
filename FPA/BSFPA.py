import numpy
import CEC2005.functions as cec2005


def distance(x, y):
    return numpy.linalg.norm(x - y)


def Weibull(i):
    """
    Weibull 分布
    """
    m = 1.5
    gamma = 1
    return (m / gamma) * ((i / gamma) ** (m - 1)) * numpy.exp(-((i / gamma) ** m))


def logistic_map(x):
    return 3.99 * x * (1 - x)


class BS_FPA:
    # Biomimetic Search FPA
    def __init__(self, num_pop, dim, ub, lb, f_obj, verbose):
        self.num_pop = num_pop  # 种群数目
        self.dim = dim  # 维数
        self.ub = ub  # 上界
        self.lb = lb  # 下界
        self.pop = numpy.empty((num_pop, dim))  # 种群
        self.f_obj = f_obj  # 目标函数
        self.f_score = numpy.empty((num_pop, 1))
        self.verbose = verbose  # 显示迭代过程
        self.p_best = None  # 最好个体
        self.f_best = None  # 最好适应度值
        self.iter_f_score = []  # 每次迭代的适应度值

    def initialize(self):
        """
        种群初始化
        :return: 初始化种群和种群分数
        """
        num_boundary = len(self.ub)
        if num_boundary == 1:
            for i in range(self.num_pop):
                self.pop[i, :] = self.lb + (self.ub - self.lb) * numpy.random.rand(1, self.dim).flatten()
                self.f_score[i] = self.f_obj(self.pop[i, :])
        else:
            logistic = numpy.random.rand(self.num_pop, self.dim)  # 初始化随机数

            '''采用 Logistic 混沌映射对花粉种群作初始化'''
            for i in range(self.num_pop - 1):
                logistic[i + 1, :] = logistic_map(logistic[i, :])
            for i in range(self.dim):
                self.pop[:, i] = self.lb[i] + (self.ub[i] - self.lb[i]) * logistic[:, i]

            for i in range(self.num_pop):
                self.f_score[i] = self.f_obj(self.pop[i, :])
        return [self.pop, self.f_score]

    def get_best(self):
        """
        获取最优解
        :return: 最优索引和最优得分
        """
        idx = numpy.argmin(self.f_score)
        self.p_best = self.pop[idx, :]
        self.f_best = numpy.min(self.f_score)
        return [self.p_best, self.f_best]

    def Levy(self, beta=1.5):
        """
        莱维飞行
        :param beta: 固定参数
        :return: 随机数
        """
        sigma = (numpy.random.gamma(1 + beta) * numpy.sin(numpy.pi * beta / 2) / (
                numpy.random.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = numpy.random.normal(0, sigma, (1, self.dim))
        v = numpy.random.normal(0, 1, (1, self.dim))
        levy = u / (numpy.abs(v) ** (1 / beta))
        return levy

    def clip(self, p):
        """
        :param p: 花粉配子
        :return: 越界处理后的花粉配子
        """
        i = p < self.lb
        p[i] = numpy.array(self.lb)[i]
        j = p > self.ub
        p[j] = numpy.array(self.ub)[j]
        return p

    def BS_FPA(self, gama=1.0, belta0=1, alpha=0.02):
        """
        :param gama: 传播介质对光的吸收系数
        :param belta0: 初始吸引度值
        :param alpha: 步长扰动因子
        :return: 更新后的种群
        """

        for i in range(self.num_pop):

            # 如果是最优个体
            if self.f_score[i] == self.f_score.min():
                # 对处在最佳位置的个体进行随机扰动
                new_pop = self.pop[i, :] + alpha * (numpy.random.normal() - 0.5)

                # 确保更新后的位置在取值范围内
                new_pop = self.clip(new_pop.flatten())
                new_f_score = self.f_obj(new_pop)

                if new_f_score < self.f_score[i]:
                    self.pop[i, :] = new_pop
                    self.f_score[i] = new_f_score

            # 其他个体的位置更新方式
            else:
                # 当前个体被其他适应度值优于自身的个体所吸引，然后进行位置移动
                for j in range(self.num_pop):
                    if self.f_score[j] < self.f_score[i]:
                        # 更新种群，向最亮处靠拢
                        belta = belta0 * numpy.exp((-gama) * (distance(self.pop[j, :], self.pop[i, :]) ** 2))
                        new_pop = self.pop[j, :] + belta * (self.pop[i, :] - self.pop[j, :]) + alpha * (
                            numpy.random.normal())

                        # 确保更新后的位置在取值范围内
                        new_pop = self.clip(new_pop.flatten())
                        new_f_score = self.f_obj(new_pop)

                        if new_f_score < self.f_score[j]:
                            self.pop[j, :] = new_pop
                            self.f_score[j] = new_f_score

                        if new_f_score < self.f_best:
                            self.p_best = new_pop
                            self.f_best = new_f_score

            # 更新种群历史最优解以及对应的适应度值
            if self.f_score[i] < self.f_best:
                self.f_best = self.f_score[i]
                self.p_best = self.pop[i]

            # 重新计算每个个体的适应度函数值
            for n in range(self.num_pop):
                self.f_score[n] = self.f_obj(self.pop[n, :])

        return self.pop

    def fit(self, max_generation, function_number):
        """
        算法迭代
        :param max_generation: 最大迭代次数
        :param p: 切换概率
        :return: 最优值和最优个体
        """
        self.pop, self.f_score = self.initialize()  # 初始化
        self.p_best, self.f_best = self.get_best()  # 当前最优个体

        self.iter_f_score.append(self.f_best)
        self.pop = self.BS_FPA()

        problem = getattr(cec2005, "problem" + str(function_number))

        # 定义收敛精度
        convergence_threshold = problem['best'] + 1e-5

        # 定义最小迭代次数
        min_epoch = 999999  # 初始化为正无穷，确保第一次达到收敛时会更新

        converged = False  # 是否达到收敛

        # 迭代
        for i in range(max_generation):

            '''将韦伯分布函数和迭代次数结合以实现动态控制转换概率'''
            p = 0.6 - 0.1 * Weibull(i) * (1 - i / max_generation)

            '''使用萤火虫算法更新种群'''
            # self.pop = self.BS_FPA()

            self.p_best, self.f_best = self.get_best()  # 重新计算每个个体的适应度函数值

            for j in range(self.num_pop):

                if numpy.random.rand() < p:
                    # 异花授粉公式
                    # VS = abs((self.f_best - self.f_score[j]) / max(self.f_best, self.f_score[j]))

                    '''自适应步长缩放'''
                    theta = 0.2 * numpy.exp(- i / max_generation)

                    '''全局随机变异'''
                    idx_set = numpy.random.choice(range(self.num_pop), 2)
                    vs = 0.01 * self.pop[idx_set[0]] + 0.1 * (self.pop[idx_set[0]] - self.pop[idx_set[1]])

                    new_pop = self.pop[j, :] + theta * self.Levy() * (self.pop[j, :] - self.p_best) + vs
                else:
                    # 自花授粉公式
                    idx_set = numpy.random.choice(range(self.num_pop), 2)

                    '''引入交叉算子'''
                    # theta = numpy.random.uniform(0, 1)
                    # new_pop = theta * self.pop[j, :] + (1 - theta) * (self.pop[j, :] - self.pop[idx_set[0], :])

                    '''基于瑞利分布的繁衍概率'''
                    U = 0.3 - (i / 4) * numpy.exp((-i ** 2) / 8) * ((-i / max_generation) ** 30)

                    '''通过柯西变异进行随机扰动，以增加种群多样性避免陷入局部最优'''
                    cauchy = ((max_generation - i) / max_generation) * numpy.tan(numpy.pi * (numpy.random.rand() - 0.5))

                    new_pop = self.pop[j, :] * cauchy + U * numpy.random.rand(1, self.dim) * (
                            self.pop[idx_set[0], :] - self.pop[idx_set[1], :])

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

            difference = abs(self.f_best - convergence_threshold)
            if difference < 1e-5 and i < min_epoch:
                converged = True
                min_epoch = i

            self.iter_f_score.append(self.f_best)

            if self.verbose:
                print("============{}/{}==============".format(i + 1, max_generation))
                print(self.f_best)

        if converged:
            print("BSFPA 达到收敛精度！最小迭代次数：", min_epoch, " 最优解：", self.f_best)
        else:
            print("BSFPA 未达到收敛精度！最优解：", self.f_best)
        return [self.iter_f_score, self.f_best]
