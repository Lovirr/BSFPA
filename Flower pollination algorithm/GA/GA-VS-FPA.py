import numpy as np
from matplotlib import pyplot as plt
from mealpy.evolutionary_based import FPA, GA
import os
import time

# 定义目标函数
def fitness_function(x):
    dim = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    sum_1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / dim))
    sum_2 = np.exp(np.sum(np.cos(c * x)) / dim)
    y = sum_1 - sum_2 + a + np.exp(1)
    return y


problem = {
    "fit_func": fitness_function,
    "lb": [-32, ] * 30,
    "ub": [32, ] * 30,
    "minmax": "min",
    "log_to": None,
    "save_population": False,
}

epoch = 1000  # 最大迭代次数
pop_size = 50  # 种群数量

## 运行
time_start = time.time()
ga_model = GA.BaseGA(epoch, pop_size, pc=0.9, pm=0.05)
ga_best_x, ga_best_f = ga_model.solve(problem)
time_end = time.time()
print(f"GA Best solution: {ga_best_x}\nGA Best fitness: {ga_best_f}")
print(f"GA Time cost: {time_end - time_start}s")

time_start = time.time()
fpa_model = FPA.OriginalFPA(epoch, pop_size)
fpa_best_x, fpa_best_f = fpa_model.solve(problem)
time_end = time.time()
print(f"FPA Best solution: {fpa_best_x}\nFPA Best fitness: {fpa_best_f}")
print(f"FPA Time cost: {time_end - time_start}s")

''' 输出结果 '''
# 绘制适应度曲线
plt.figure(figsize=(8, 6), dpi=300)

plt.plot(ga_model.history.list_global_best_fit, 'r-', linewidth=2, label='GA')
plt.plot(fpa_model.history.list_global_best_fit, 'g-', linewidth=2, label='FPA')

plt.title('Convergence curve: ', fontsize=15)
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.grid()

plt.legend()
plt.show()

# 播放提示音
os.system('afplay /Users/lovir/Music/三全音.aif')
