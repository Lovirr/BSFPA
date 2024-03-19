import numpy as np
import matplotlib.pyplot as plt
from mealpy.swarm_based import PSO
from models.readtsp import read
from models.tsp_model import TravellingSalesmanProblem
from models.tsp_solution import generate_stable_solution

filename = "chn31"
DATA_POS = read(f"TSPLIB/{filename}.tsp")

DATA_POS = np.array(DATA_POS)
N_CITIES = DATA_POS.shape[0]
TSP = TravellingSalesmanProblem(n_cities=N_CITIES, city_positions=DATA_POS)
TSP.plot_cities(pathsave="./results/WOA-TSP", filename="cities_map")

LB = [0, ] * TSP.n_cities
UB = [(TSP.n_cities - 0.01), ] * TSP.n_cities

problem = {
    "fit_func": TSP.fitness_function,
    "lb": LB,
    "ub": UB,
    "minmax": "min",        # Trying to find the minimum distance
    "log_to": "console",
    "amend_position": generate_stable_solution
}
# 最大迭代次数
epoch = 2000

# 种群数量
pop_size = 50

model = PSO.OriginalPSO(epoch)
best_position, best_fitness = model.solve(problem)
print(f"Best solution: {best_position}\nObj = Total Distance: {best_fitness}")

# 绘制收敛曲线
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(model.history.list_global_best_fit, 'black', linewidth=2, label='PSO')
plt.title('PSO TSP Convergence curve: ', fontsize=15)
plt.xlabel('Iteration')  # 设置x轴标签
plt.ylabel('Fitness')  # 设置y轴标签
plt.grid()  # 显示网格
plt.legend()  # 显示图例
plt.show()  # 显示图像

dict_solutions = {}
for idx, g_best in enumerate(model.history.list_global_best):
    dict_solutions[idx] = [g_best[0], g_best[1][0]]  # Final solution and fitness
    
# 保存动画
# TSP.plot_animate(dict_solutions, filename="PSO-TSP-results", pathsave="./results/PSO-TSP")

# 保存图片
TSP.plot_solutions(dict_solutions, filename=f"PSO-{filename}-epochs", pathsave="./results/PSO-TSP")

