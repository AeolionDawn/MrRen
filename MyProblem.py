
# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

"""
该案例展示了一个带约束连续决策变量的最小化目标的双目标优化问题。
min f1 = X**2
min f2 = (X - 2)**2
s.t.
X**2 - 2.5 * X + 1.5 >= 0
10 <= Xi <= 10, (i = 1,2,3,...)
"""

class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self, M, model, sample, target_label):
        self.model = model
        self.sample = sample
        self.target_label = target_label

        name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
        maxormins = [1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）

        Dim = M # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        # lb = [-10] * Dim # 决策变量下界
        # ub = [10] * Dim # 决策变量上界
        lb = [0] * Dim
        ub = [0] * Dim
        for i in range(M):  # 10 30 50
            lb[i] = max(sample[i] - 80 / 255, 0.0)
            ub[i] = min(sample[i] + 80 / 255, 1.0)

        lbin = [1] * Dim # 决策变量下边界
        ubin = [1] * Dim # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop): # 目标函数
#         Vars = pop.Phen # 得到决策变量矩阵
#         f1 = Vars**2
#         f2 = (Vars - 2)**2
# #        # 利用罚函数法处理约束条件
# #        exIdx = np.where(Vars**2 - 2.5 * Vars + 1.5 < 0)[0] # 获取不满足约束条件的个体在种群中的下标
# #        f1[exIdx] = f1[exIdx] + np.max(f1) - np.min(f1)
# #        f2[exIdx] = f2[exIdx] + np.max(f2) - np.min(f2)
#         # 利用可行性法则处理约束条件

#         pop.ObjV = np.hstack([f1, f2]) # 把求得的目标函数值赋值给种群pop的ObjV
        # Vars = pop.Phen # Get the decision variables
        pred = self.model.predict(pop.Phen)[:, [self.target_label]]
        threshold = 1 / 10
        pred[pred < threshold] = threshold
        dist = np.linalg.norm(self.sample - pop.Phen, ord=2, axis=1)
        pop.ObjV = np.hstack([pred, dist.reshape(dist.shape[0], 1)]) # 把求得的目标函数值赋值给种群pop的ObjV

