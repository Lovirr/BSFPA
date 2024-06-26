import time

from matplotlib import pyplot as plt
from mealpy import GA

import CEC2005.functions as cec2005

# 定义目标函数
fitness_function = cec2005.fun1

# 维度
dim = 30

# 搜索空间范围
bounds = 100

# 最大迭代次数
epoch = 1000

# 种群数量
pop_size = 50

problem = {
    "fit_func": fitness_function,
    "lb": [-bounds, ] * dim,
    "ub": [bounds, ] * dim,
    "minmax": "min",
}

'''GA'''
time_start = time.time()
ga_model = GA.BaseGA(epoch, pop_size, pc=0.9, pm=0.05)
ga_best_x, ga_best_f = ga_model.solve(problem)
time_end = time.time()
ga_cost = time_end - time_start

''' 打印结果 '''
print("----------Best fitness----------")
print(f"GA Best fitness: {ga_best_f}")

print("----------Time cost----------")
print(f"GA Time cost: {ga_cost}s")

''' 输出结果 '''
function_name = fitness_function.__name__
filename = f"{'CEC2005 ' + function_name}.jpg"
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(ga_model.history.list_global_best_fit, 'r', linewidth=2, label='GA')
plt.title(f'CEC2005 ' + function_name + ' Convergence curve: ', fontsize=15)
plt.xlabel('Iteration')  # 设置x轴标签
plt.ylabel('Fitness')  # 设置y轴标签
plt.grid()  # 显示网格
plt.legend()  # 显示图例
plt.show()  # 显示图像
