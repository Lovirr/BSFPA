# 库的导入
import numpy as np
import matplotlib.pyplot as plt
import CEC2005.functions as cec2005

# 可以更改为 2, 3, ..., 23 来调用其他函数
function_number = 1

# 构建函数名称
function_name = "fun" + str(function_number)

# 通过 getattr() 函数获取函数对象
function = getattr(cec2005, function_name)
problem = getattr(cec2005, "problem" + str(function_number))


# 计算个体间的空间距离
def distance(x, y):
    return np.linalg.norm(x - y)


dim = problem["dim"]  # 维度
lb = problem["lb"]  # 下界
ub = problem["ub"]  # 上界
rangepop = [-30, 30]  # 取值范围
n_pop = 50  # 种群数量
gama = 1.0  # 传播介质对光的吸收系数
belta0 = 1.0  # 初始吸引度值
alpha = 1  # 步长扰动因子
epoch = 1000  # 迭代次数

# pop用于存储种群个体的位置信息，fitness用于存储个体对应的适应度值
pop = np.zeros((n_pop, dim))
fitness = np.zeros(n_pop)

# 对种群个体进行初始化并计算对应适应度值
for j in range(n_pop):
    pop[j, :] = np.random.uniform(lb, ub, (1, dim))
    fitness[j] = function(pop[j])

# bestpop，bestfit分别表示种群历史最优解和适应度值
bestpop, bestfit = pop[fitness.argmin()].copy(), fitness.min()

# bestfitness用于存储每次迭代时的种群历史最优适应度值
bestfitness = np.zeros(epoch)

# 开始训练
for i in range(epoch):
    print("============{}/{}==============".format(i + 1, epoch))
    # 对每个个体位置与适应度值进行更新
    for j in range(n_pop):
        # 当前个体适应度值为种群最优的位置更新方式
        if fitness[j] == fitness.min():
            pop[j] = pop[j] + alpha * (np.random.rand() - 0.5)
            # 确保更新后的位置在取值范围内
            pop[j][pop[j] < rangepop[0]] = rangepop[0]
            pop[j][pop[j] > rangepop[1]] = rangepop[1]
            fitness[j] = function(pop[j])
        # 其他个体的位置更新方式
        else:
            # 当前个体被其他适应度值优于自身的个体所吸引，然后进行位置移动
            for q in range(n_pop):
                if fitness[q] < fitness[j]:
                    d = distance(pop[j], pop[q])
                    belta = belta0 * np.exp((-gama) * (d ** 2))
                    pop[j, :] = pop[j, :] + belta * (pop[q, :] - pop[j, :]) + alpha * (np.random.rand() - 0.5)
                    # 确保更新后的位置在取值范围内
                    pop[j][pop[j] < rangepop[0]] = rangepop[0]
                    pop[j][pop[j] > rangepop[1]] = rangepop[1]
                    fitness[j] = function(pop[j])
        # 更新种群历史最优解以及对应的适应度值
        if fitness[j] < bestfit:
            bestfit = fitness[j]
            bestpop = pop[j]
    # 存储当前迭代时的种群历史最优适应度值
    bestfitness[i] = bestfit
    print(bestfitness[i])

# print("After iteration, the best individual is:", bestpop)
print("the best fitness is:", bestfit)
# 输出训练后种群个体适应度值的均值与标准差
mean = np.sum(fitness) / n_pop
std = np.std(fitness)
# print("the mean fitness of the swarm is:", "%e" % mean)
# print("the std fitness of the swarm is:", "%e" % std)

# 将结果进行绘图
fig = plt.figure(figsize=(8, 6), dpi=300)
plt.title(f'CEC2005 ' + function_name + ' Convergence curve: ', fontsize=15)
x = range(1, epoch + 1, 1)
plt.plot(x, bestfitness, color="red", label="FA", linewidth=2.0)
plt.tick_params(labelsize=15)
plt.xlim(0, epoch + 1)
plt.yscale("log")
plt.xlabel("Iteration", fontdict={'weight': 'normal', 'size': 15})
plt.ylabel("Fitness", fontdict={'weight': 'normal', 'size': 15})
plt.xticks(range(0, epoch + 1, int(epoch / 10)))
plt.legend()  # 显示图例
plt.show()
