import random
import numpy as np
import matplotlib.pyplot as plt

# 定义旅行商问题的参数
num_cities = 50  # 城市数量
num_flowers = 50  # 花朵数量
max_iterations = 1000  # 最大迭代次数

# 生成随机的城市坐标
cities = np.random.rand(num_cities, 2)

# 初始化种群
population = []
for _ in range(num_flowers):
    solution = list(range(num_cities))
    random.shuffle(solution)
    population.append(solution)

# 计算路径长度
def calculate_distance(solution):
    distance = 0
    for i in range(num_cities-1):
        city1 = solution[i]
        city2 = solution[i+1]
        distance += np.linalg.norm(cities[city1] - cities[city2])
    distance += np.linalg.norm(cities[solution[-1]] - cities[solution[0]])  # 回到起始城市
    return distance

# 主循环
best_solution = None
best_distance = float('inf')
iteration = 0

while iteration < max_iterations:
    # 花朵授粉
    for i in range(num_flowers):
        current_solution = population[i]

        # 随机选择一个邻域解
        neighbor = current_solution.copy()
        city1, city2 = random.sample(range(num_cities), 2)
        neighbor[city1], neighbor[city2] = neighbor[city2], neighbor[city1]

        # 计算适应度
        current_distance = calculate_distance(current_solution)
        neighbor_distance = calculate_distance(neighbor)

        # 更新最优解
        if neighbor_distance < best_distance:
            best_solution = neighbor.copy()
            best_distance = neighbor_distance

        # 更新种群
        if neighbor_distance < current_distance:
            population[i] = neighbor.copy()

    iteration += 1

# 输出结果
print("最优路径：", best_solution)
print("最短路径长度：", best_distance)

# 绘制城市坐标和标注
x = cities[:, 0]
y = cities[:, 1]
plt.figure(figsize=(8, 6), dpi=300)
for i in range(num_cities):
    plt.text(x[i], y[i], str(i+1), fontsize=8, color='black', ha='center', va='bottom')
plt.scatter(x, y, color='b')

# 绘制最优路径
best_x = x[best_solution + [best_solution[0]]]
best_y = y[best_solution + [best_solution[0]]]
plt.plot(best_x, best_y, color='r', linewidth=1.5)

# 添加标签和图例
plt.title('Traveling Salesman Problem')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图像
plt.show()