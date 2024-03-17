from mealpy.swarm_based import PSO, ACOR, FA
from mealpy.evolutionary_based import FPA, GA
from matplotlib import pyplot as plt
import CEC2005.functions as cec2005
import os
import time

# 定义目标函数
fitness_function = cec2005.fun5

# 维度
dim = 30

# 搜索空间范围
bounds = 30

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

## 运行
# i = 0
# for (i, fitness_function) in enumerate(fitness_functions):
#     title = fitness_functions[i].__name__.replace('.', '_')
time_start = time.time()
acor_model = ACOR.OriginalACOR(epoch, pop_size, sample_count=25, intent_factor=0.5, zeta=1.0)
acor_best_x, acor_best_f = acor_model.solve(problem)
time_end = time.time()
acor_cost = time_end - time_start

time_start = time.time()
fpa_model = FPA.OriginalFPA(epoch, pop_size)
fpa_best_x, fpa_best_f = fpa_model.solve(problem)
time_end = time.time()
fpa_cost = time_end - time_start

time_start = time.time()
pso_model = PSO.OriginalPSO(epoch, pop_size)
pso_best_x, pso_best_f = pso_model.solve(problem)
time_end = time.time()
pso_cost = time_end - time_start

time_start = time.time()
ga_model = GA.BaseGA(epoch, pop_size, pc=0.9, pm=0.05)
ga_best_x, ga_best_f = ga_model.solve(problem)
time_end = time.time()
ga_cost = time_end - time_start

time_start = time.time()
fa_model = FA.OriginalFA(epoch, pop_size, max_sparks=50, p_a=0.04, p_b=0.8, max_ea=40, m_sparks=50)
fa_best_x, fa_best_f = fa_model.solve(problem)
time_end = time.time()
fa_cost = time_end - time_start

print("----------Best fitness----------")
print(f"ACOR Best fitness: {acor_best_f}")
print(f"FPA Best fitness: {fpa_best_f}")
print(f"PSO Best fitness: {pso_best_f}")
print(f"GA Best fitness: {ga_best_f}")
print(f"FA Best fitness: {fa_best_f}")

print("----------Time cost----------")
print(f"ACOR Time cost: {acor_cost}s")
print(f"FPA Time cost: {fpa_cost}s")
print(f"PSO Time cost: {pso_cost}s")
print(f"GA Time cost: {ga_cost}s")
print(f"FA Time cost: {fa_cost}s")

''' 输出结果 '''
# 绘制适应度曲线
function_name = fitness_function.__name__
filename = f"{'CEC2005 ' + function_name}.jpg"
plt.figure(figsize=(8, 6), dpi=300)


plt.plot(acor_model.history.list_global_best_fit, 'b', linewidth=2, label='ACOR')
plt.plot(fpa_model.history.list_global_best_fit, 'g', linewidth=2, label='FPA')
plt.plot(pso_model.history.list_global_best_fit, 'y', linewidth=2, label='PSO')
plt.plot(ga_model.history.list_global_best_fit, 'r', linewidth=2, label='GA')
plt.plot(fa_model.history.list_global_best_fit, 'black', linewidth=2, label='FA')

# plt.title('Convergence curve: ', fontsize=15)
plt.title(f'CEC2005 ' + function_name + ' Convergence curve: ', fontsize=15)
plt.xlabel('Iteration') # 设置x轴标签
plt.ylabel('Fitness') # 设置y轴标签

plt.grid() # 显示网格
plt.legend() # 显示图例
plt.savefig(filename) # 保存图像
plt.show()

# 播放提示音
os.system('afplay /Users/lovir/Music/三全音.aif')
