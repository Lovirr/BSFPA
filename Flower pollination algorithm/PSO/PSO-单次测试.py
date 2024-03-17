import os
import time

from matplotlib import pyplot as plt
from mealpy import PSO

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

'''PSO'''
time_start = time.time()
pso_model = PSO.OriginalPSO(epoch, pop_size)
pso_best_x, pso_best_f = pso_model.solve(problem)
time_end = time.time()
pso_cost = time_end - time_start

''' 打印结果 '''
print("----------Best fitness----------")
print(f"PSO Best fitness: {pso_best_f}")

print("----------Time cost----------")
print(f"PSO Time cost: {pso_cost}s")

''' 输出结果 '''
function_name = fitness_function.__name__
filename = f"{'CEC2005 ' + function_name}.jpg"
plt.figure(figsize=(8, 6), dpi=300)

plt.plot(pso_model.history.list_global_best_fit, 'y', linewidth=2, label='PSO')

plt.title(f'CEC2005 ' + function_name + ' Convergence curve: ', fontsize=15)
plt.xlabel('Iteration')  # 设置x轴标签
plt.ylabel('Fitness')  # 设置y轴标签

plt.grid()  # 显示网格
plt.legend()  # 显示图例
# plt.savefig(filename)  # 保存图像
plt.show()  # 显示图像

# 播放提示音
os.system('afplay /Users/lovir/Music/三全音.aif')
