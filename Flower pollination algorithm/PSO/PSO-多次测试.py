from mealpy import FPA, PSO
from matplotlib import pyplot as plt
import CEC2005.functions as cec2005
import os
import time
from openpyxl import load_workbook

# 定义目标函数
fitness_function = cec2005.fun1

# 维度
dim = 30

# 搜索空间范围
bounds = 100

# 最大迭代次数
epoch = 4000

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
i = 0
k = 16
for i in range(5):
    pso_model = PSO.OriginalPSO(epoch, pop_size)
    pso_best_x, pso_best_f = pso_model.solve(problem)
    print(f"PSO Best fitness: {pso_best_f}")

    ''' 输出结果 '''
    # 绘制适应度曲线
    function_name = fitness_function.__name__
    filename = f"{'CEC2005 ' + function_name +'_'+ str(k)}.jpg"
    plt.figure(figsize=(8, 6), dpi=300)

    plt.plot(pso_model.history.list_global_best_fit, 'b', linewidth=2, label='PSO')

    plt.title(f'CEC2005 ' + function_name + ' Convergence curve: ', fontsize=15)
    plt.xlabel('Iteration')  # 设置x轴标签
    plt.ylabel('Fitness')  # 设置y轴标签

    plt.grid()  # 显示网格
    plt.legend()  # 显示图例
    plt.savefig(filename)  # 保存图像
    plt.show()
    #
    # 打开现有的工作簿
    workbook = load_workbook('testdata.xlsx')
    # 选择指定的工作表
    sheet = workbook['f1']

    # 将数据写入指定位置
    sheet['C' + str(k)] = pso_best_f

    # 保存工作簿
    workbook.save('testdata.xlsx')



    i = i + 1
    k = k + 1
# 播放提示音
os.system('afplay /Users/lovir/Music/三全音.aif')