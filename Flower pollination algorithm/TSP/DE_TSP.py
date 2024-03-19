from mealpy.evolutionary_based import DE
from models.readtsp import read
from models.tsp_model import TravellingSalesmanProblem
from models.tsp_solution import generate_unstable_solution

import numpy as np
import matplotlib.pyplot as plt

filename = "ulysses16"
DATA_POS = read(f"TSPLIB/{filename}.tsp")

DATA_POS = np.array(DATA_POS)
N_CITIES = DATA_POS.shape[0]
TSP = TravellingSalesmanProblem(n_cities=N_CITIES, city_positions=DATA_POS)
TSP.plot_cities(pathsave="./results/DE-TSP", filename="cities_map")

LB = [1, ] * (N_CITIES - 1)
UB = [(N_CITIES - 0.01), ] * (N_CITIES - 1)

## Define problem dictionary
problem = {
    "fit_func": TSP.fitness_function,
    "lb": LB,
    "ub": UB,
    "minmax": "min",
    "amend_position": generate_unstable_solution
}

# 最大迭代次数
epoch = 2000

## Run the algorithm
model = DE.SHADE(epoch)
best_position, best_fitness = model.solve(problem)
print(f"Best position: {best_position}, Best fit: {best_fitness}")

# 绘制收敛曲线
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(model.history.list_global_best_fit, 'r', linewidth=2, label='DE')
plt.title('DE TSP Convergence curve: ', fontsize=15)
plt.xlabel('Iteration')  # 设置x轴标签
plt.ylabel('Fitness')  # 设置y轴标签
plt.grid()  # 显示网格
plt.legend()  # 显示图例
plt.show()  # 显示图像

## Show the results
dict_solutions = {}
for idx, g_best in enumerate(model.history.list_global_best):
    dict_solutions[idx] = [g_best[0], g_best[1][0]]  # Final solution and fitness

# 保存动画
# TSP.plot_animate(dict_solutions, filename="DE-TSP-results", pathsave="./results/DE-TSP")

# 保存图片
TSP.plot_solutions(dict_solutions, filename=f"DE-{filename}-epochs", pathsave="./results/DE-TSP")
