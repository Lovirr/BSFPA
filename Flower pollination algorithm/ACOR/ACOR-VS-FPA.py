from mealpy.swarm_based import ACOR
from mealpy.evolutionary_based import FPA
from matplotlib import pyplot as plt
import CEC2005.functions as cec2005
import os
import time


# 定义目标函数
fitness_function = cec2005.fun3
# fitness_functions = [cec2005.fun1, cec2005.fun2, cec2005.fun3]

dim = 30

problem = {
    "fit_func": fitness_function,
    "lb": [-100, ] * dim,
    "ub": [100, ] * dim,
    "minmax": "min",
    "log_to": None,
    "save_population": False,
}

epoch = 1000  # 最大迭代次数
pop_size = 50  # 种群数量

## 运行
# i = 0
# for (i, fitness_function) in enumerate(fitness_functions):
#     title = fitness_functions[i].__name__.replace('.', '_')
time_start = time.time()
asor_model = ACOR.OriginalACOR(epoch, pop_size, sample_count=25, intent_factor=0.5, zeta=1.0)
asor_best_x, asor_best_f = asor_model.solve(problem)
time_end = time.time()
print(f"ACOR Best solution: {asor_best_x}\nACOR Best fitness: {asor_best_f}")
print(f"ACOR Time cost: {time_end - time_start}s")

time_start = time.time()
fpa_model = FPA.OriginalFPA(epoch, pop_size)
fpa_best_x, fpa_best_f = fpa_model.solve(problem)
time_end = time.time()
print(f"FPA Best solution: {fpa_best_x}\nFPA Best fitness: {fpa_best_f}")
print(f"FPA Time cost: {time_end - time_start}s")

''' 输出结果 '''
# 绘制适应度曲线
plt.figure(figsize=(8, 6), dpi=300)

plt.plot(asor_model.history.list_global_best_fit, 'black', linewidth=2, label='ACOR')
plt.plot(fpa_model.history.list_global_best_fit, 'g-', linewidth=2, label='FPA')

# plt.title('Convergence curve: ', fontsize=15)
plt.title(f'CEC2005  Convergence curve: ', fontsize=15)
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.grid()

plt.legend()
plt.show()

# 播放提示音
os.system('afplay /Users/lovir/Music/三全音.aif')
