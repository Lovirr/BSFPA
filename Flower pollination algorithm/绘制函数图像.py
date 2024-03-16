import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import CEC2005.functions as cec2005

# 定义目标函数
fitness_functions = [cec2005.fun1, cec2005.fun2, cec2005.fun3, cec2005.fun4, cec2005.fun5,
                     cec2005.fun6, cec2005.fun7, cec2005.fun8, cec2005.fun9, cec2005.fun10,
                     cec2005.fun11, cec2005.fun12, cec2005.fun13, cec2005.fun14,
                     cec2005.fun16, cec2005.fun17, cec2005.fun18, ]

def plot_surface(fun, x_range, y_range, filename):
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fun(np.array([X[i, j], Y[i, j]]))  # Convert x to NumPy array
    fig = plt.figure(figsize=(8, 6), dpi=300)  # 设置图像大小和分辨率
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(filename)  # 保存图像
    plt.close(fig)  # 关闭图像窗口

x_range = np.linspace(-100, 100, 100)
y_range = np.linspace(-100, 100, 100)

for i, fun in enumerate(fitness_functions):
    print(f"Plotting function {i + 1}/{len(fitness_functions)}")
    # 获取函数名称，并将非字母数字字符替换为下划线
    function_name = fun.__name__.replace('.', '_')
    filename = f"{function_name}.png"
    plot_surface(fun, x_range, y_range, filename)
