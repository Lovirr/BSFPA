import opfunu
import numpy as np
from matplotlib import pyplot as plt
from mealpy.swarm_based import WOA, GWO, PSO, ACOR, FA, ACOR
from mealpy.evolutionary_based import FPA, GA
import os

fun_name = 'F8'  # 按需修改
year = '2005'  # 按需修改
func_num = fun_name + year
dim = 30  # 维度，根据cec函数 选择对应维度
epoch = 100  # 最大迭代次数
pop_size = 50  # 种群数量
'''定义的 cec函数 '''


def cec_fun(x):
    funcs = opfunu.get_functions_by_classname(func_num)
    func = funcs[0](ndim=dim)
    F = func.evaluate(x)
    return F


''' fit_func->目标函数, lb->下限, ub->上限 '''
bounds = 500
problem_dict = {
    "fit_func": cec_fun,
    "lb": [-bounds, ] * dim,
    "ub": [bounds, ] * dim,
    "minmax": "min",
}

''' 调用优化算法 '''

acor_model = ACOR.OriginalACOR(epoch, pop_size, sample_count=25, intent_factor=0.5, zeta=1.0)
fpa_model = FPA.OriginalFPA(epoch, pop_size)
ga_model = GA.BaseGA(epoch, pop_size, pc=0.9, pm=0.05)
pso_model = PSO.OriginalPSO(epoch, pop_size)
fa_model = FA.OriginalFA(epoch, pop_size, max_sparks=50, p_a=0.04, p_b=0.8, max_ea=40, m_sparks=50)

'''求解 cec函数 '''
acor_best_x, acor_best_f = acor_model.solve(problem_dict)
fpa_best_x, fpa_best_f = fpa_model.solve(problem_dict)
ga_best_x, ga_best_f = ga_model.solve(problem_dict)
pso_best_x, pso_best_f = pso_model.solve(problem_dict)
fa_best_x, fa_best_f = fa_model.solve(problem_dict)

print("----------Best fitness----------")
print(f"ACOR Best fitness: {acor_best_f}")
print(f"FPA Best fitness: {fpa_best_f}")
print(f"GA Best fitness: {ga_best_f}")
print(f"PSO Best fitness: {pso_best_f}")
print(f"FA Best fitness: {fa_best_f}")

''' 
    绘制适应度曲线
    model.history.list_global_best_fit：适应度曲线
'''
plt.figure(figsize=(8, 6), dpi=300)
step=epoch/10

plt.plot(acor_model.history.list_global_best_fit, 'r-', marker='o', markevery=10, linewidth=1, label='ACOR')
plt.plot(fpa_model.history.list_global_best_fit, 'g-', marker='x', markevery=10, linewidth=1, label='FPA')
plt.plot(ga_model.history.list_global_best_fit, 'y', marker='D', markevery=10, linewidth=1, label='GA')
plt.plot(pso_model.history.list_global_best_fit, 'black', marker='*', markevery=10, linewidth=2, label='PSO')
plt.plot(fa_model.history.list_global_best_fit, 'b', linewidth=2, marker='s', markevery=step, label='FA')

plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.grid()
plt.title('Convergence curve: ' + 'CEC' + year + '-' + fun_name + ', Dim=' + str(dim))
plt.legend()
plt.show()

# 播放提示音
os.system('afplay /Users/lovir/Music/三全音.aif')
