import time
import numpy
import json
from FPA import FPA
from BSFPA import BS_FPA
from matplotlib import pyplot as plt
from mealpy import PSO, GA, FA
import CEC2005.functions as cec2005

# 测试
if __name__ == "__main__":
    # 可以更改为 2, 3, ..., 20 来调用其他函数
    function_number = 1
    # 迭代次数
    epoch = 1000
    # 种群数
    n_pop = 20
    # 通过 getattr() 函数获取函数对象
    function_name = "fun" + str(function_number)
    fitness_function = getattr(cec2005, function_name)
    problem = getattr(cec2005, "problem" + str(function_number))
    dim = problem['dim']
    upper = problem['ub']
    lower = problem['lb']

    problem_dict = {
        "fit_func": fitness_function,
        "lb": problem['lb'],
        "ub": problem['ub'],
        "minmax": "min",
        "log_to": None
    }

    '''GA'''
    time_start = time.time()
    ga_model = GA.BaseGA(epoch, n_pop, pc=0.9, pm=0.05)
    ga_best_x, ga_best_f = ga_model.solve(problem_dict)
    time_end = time.time()
    ga_cost = time_end - time_start

    '''FA'''
    time_start = time.time()
    fa_model = FA.OriginalFA(epoch, n_pop, max_sparks=50, p_a=0.04, p_b=0.8, max_ea=40, m_sparks=50)
    fa_best_x, fa_best_f = fa_model.solve(problem_dict)
    time_end = time.time()
    fa_cost = time_end - time_start

    '''PSO'''
    time_start = time.time()
    pso_model = PSO.OriginalPSO(epoch, n_pop)
    pso_best_x, pso_best_f = pso_model.solve(problem_dict)
    time_end = time.time()
    pso_cost = time_end - time_start

    '''BSFPA'''
    time_start = time.time()
    bs_fpa = BS_FPA(n_pop, dim, upper, lower, fitness_function, True)
    iter_score_bs_fpa, bs_fpa_best_f = bs_fpa.fit(epoch, function_number)
    time_end = time.time()
    bs_fpa_cost = time_end - time_start
    Y_axis_BS_FPA = numpy.array(iter_score_bs_fpa)

    '''FPA'''
    time_start = time.time()
    fpa = FPA(n_pop, dim, upper, lower, fitness_function, True)
    iter_score_fpa, fpa_best_f = fpa.fit(epoch, function_number)
    time_end = time.time()
    fpa_cost = time_end - time_start
    Y_axis_FPA = numpy.array(iter_score_fpa)

    ''' 打印结果 '''
    print("----------Best fitness----------")
    print(f"BSFPA Best fitness: {bs_fpa_best_f}")
    print(f"FPA Best fitness: {fpa_best_f}")
    print(f"PSO Best fitness: {pso_best_f}")
    print(f"GA Best fitness: {ga_best_f}")
    print(f"FA Best fitness: {fa_best_f}")
    print("\n----------Time cost----------")
    print(f"BSFPA Time cost: {bs_fpa_cost}s")
    print(f"FPA Time cost: {fpa_cost}s")
    print(f"PSO Time cost: {pso_cost}s")
    print(f"GA Time cost: {ga_cost}s")
    print(f"FA Time cost: {fa_cost}s")

    # 将每个元素的格式转换为2位小数的科学记数法形式
    X_axis = [int(i) for i in range(0, epoch + 1)]
    Y_axis_BS_FPA_scientific = [format(x, '.2e') for x in numpy.around(Y_axis_BS_FPA, decimals=5)]
    Y_axis_FPA_scientific = [format(x, '.2e') for x in numpy.around(Y_axis_FPA, decimals=5)]
    Y_axis_PSO_scientific = [format(x, '.2e') for x in numpy.around(pso_model.history.list_global_best_fit, decimals=5)]
    Y_axis_GA_scientific = [format(x, '.2e') for x in numpy.around(ga_model.history.list_global_best_fit, decimals=5)]
    Y_axis_FA_scientific = [format(x, '.2e') for x in numpy.around(fa_model.history.list_global_best_fit, decimals=5)]

    function_name = "f" + str(function_number)

    # with open('data1.json', 'w') as json_file:
    #     for d1, d2, d3, d4, d5, d6 in zip(X_axis, Y_axis_BS_FPA_scientific, Y_axis_FPA_scientific,
    #                                       Y_axis_PSO_scientific, Y_axis_GA_scientific, Y_axis_FA_scientific):
    #         json.dump([function_name, d1, d2, d3, d4, d5, d6], json_file)
    #         json_file.write(',\n')
    #
    # alltime_formatted = {function_name: [f"{val:.2e}" for val in numpy.array([bs_fpa_cost, fpa_cost, pso_cost, ga_cost, fa_cost])]}
    # with open('time.json', 'a') as json_file:
    #     json.dump(alltime_formatted, json_file)
    #     json_file.write(',\n')
    #
    # best_formatted = {function_name: [f"{val:.2e}" for val in numpy.array([bs_fpa_best_f, fpa_best_f, pso_best_f, ga_best_f, fa_best_f])]}
    # with open('best.json', 'a') as json_file:
    #     json.dump(best_formatted, json_file)
    #     json_file.write(',\n')

    # worst_formatted = {function_name: [f"{val:.2e}" for val in numpy.array([bs_fpa_best_f, fpa_best_f, pso_best_f, ga_best_f, fa_best_f])]}
    # with open('worst.json', 'a') as json_file:
    #     json.dump(worst_formatted, json_file)
    #     json_file.write(',\n')

    # average_formatted = {function_name: [f"{val:.2e}" for val in numpy.array([bs_fpa_best_f, fpa_best_f, pso_best_f, ga_best_f, fa_best_f])]}
    # with open('average.json', 'a') as json_file:
    #     json.dump(average_formatted, json_file)
    #     json_file.write(',\n')
    # print(function_name+" done")

    # 画图
    plt.figure(figsize=(8, 6), dpi=300)  # 设置图片大小
    plt.title(f'CEC2005 ' + function_name + ' Convergence curve: ', fontsize=15)  # 设置标题
    plt.plot(X_axis, Y_axis_BS_FPA[X_axis], 'b', linewidth=2, label='BSFPA')
    plt.plot(X_axis, Y_axis_FPA[X_axis], 'g', linewidth=2, label='FPA')
    plt.plot(pso_model.history.list_global_best_fit, 'y', linewidth=2, label='PSO')
    plt.plot(ga_model.history.list_global_best_fit, 'r', linewidth=2, label='GA')
    plt.plot(fa_model.history.list_global_best_fit, 'black', linewidth=2, label='FA')
    plt.xlabel('Iteration')  # 设置x轴标签
    plt.ylabel('Fitness')  # 设置y轴标签
    # plt.yscale("log") # 设置y轴为对数坐标
    plt.legend()  # 显示图例
    plt.show()
