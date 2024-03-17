from mealpy.swarm_based import PSO
from mealpy.evolutionary_based import FPA
from matplotlib import pyplot as plt
import CEC2005.functions as cec2005
import os
import time


# 定义目标函数
# fitness_function = cec2005.fun22
fitness_functions = [cec2005.fun3]

dim = 30
i = 0
problem = {
    "fit_func": fitness_functions[i],
    "lb": [-100, ] * dim,
    "ub": [100, ] * dim,
    "minmax": "min",
    "log_to": None,
    "save_population": False,
}

epoch = 1000  # 最大迭代次数
pop_size = 50  # 种群数量

## 运行

for (i, fitness_function) in enumerate(fitness_functions):
    title = fitness_functions[i].__name__.replace('.', '_')
    time_start = time.time()
    pso_model = PSO.OriginalPSO(epoch, pop_size)
    pso_best_x, pso_best_f = pso_model.solve(problem)
    time_end = time.time()
    print(f"PSO Best solution: {pso_best_x}\nPSO Best fitness: {pso_best_f}")
    print(f"PSO Time cost: {time_end - time_start}s")

    time_start = time.time()
    fpa_model = FPA.OriginalFPA(epoch, pop_size)
    fpa_best_x, fpa_best_f = fpa_model.solve(problem)
    time_end = time.time()
    print(f"FPA Best solution: {fpa_best_x}\nFPA Best fitness: {fpa_best_f}")
    print(f"FPA Time cost: {time_end - time_start}s")

    ''' 输出结果 '''
    # 绘制适应度曲线
    plt.figure(figsize=(8, 6), dpi=300)

    plt.plot(pso_model.history.list_global_best_fit, 'black', linewidth=2, label='PSO')
    plt.plot(fpa_model.history.list_global_best_fit, 'g-', linewidth=2, label='FPA')

    # plt.title('Convergence curve: ', fontsize=15)
    plt.title(f'CEC2005 {title} Convergence curve: ', fontsize=15)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid()

    plt.legend()
    plt.show()

    # 播放提示音
    os.system('afplay /Users/lovir/Music/三全音.aif')
