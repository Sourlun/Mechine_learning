# NumPy支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
import numpy as np
# matplotlib 是 Python 的绘图库。
# 它可与 NumPy 一起使用，提供了一种有效的 MatLab 开源替代方案。 它也可以和图形工具包一起使用
import matplotlib.pyplot as plt

# 导入数据
points = np.genfromtxt('../data/training/data.csv', delimiter=',')  # delimiter:分隔符
## 变成二维数组输出
# print(points)

# 获取x,y轴的数据
## [:全部,每一项数组的第几个]
x = points[:, 0]
y = points[:, 1]


# 定义损失函数
## 损失函数是系数的函数, 另外还有传入x, y
def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)

    # 逐点计算平方损失误差, 然后求平均数
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        # 公式参考资料:求解线性回归.png
        total_cost = (y - w * x - b) ** 2 + total_cost

    return total_cost / M



# Scikit-learn(sklearn)是机器学习中常用的第三方模块，对常用的机器学习方法进行了封装，
# 包括回归(Regression)、降维(Dimensionality Reduction)、分类(Classfication)、聚类(Clustering)等方法。
# 当我们面临机器学习问题时，便可根据下图来选择相应的方法。Sklearn具有以下特点：
# 简单高效的数据挖掘和数据分析工具
# 让每个人能够在复杂环境中重复使用
# 建立NumPy、Scipy、MatPlotLib之上
from sklearn.linear_model import LinearRegression   # LinearRegression:线性回归

# 转成二维数组
x_new = x.reshape(-1, 1) # reshape()是数组array中的方法，作用是将数据重新组织 (生成x行y列矩阵, 这里是不限定多少行一列的二维数组)
y_new = y.reshape(-1, 1)
print(' ------------------ reshape后的x结果: --------------------')
print(x_new)
print(' --------------------reshape后的y结果: -------------------------')
print(y_new)


lr = LinearRegression() # 创建一个线性回归模型对象
# 拟合 (传参是个二维数组)
lr.fit(x_new, y_new)

# 从训练好的模型中提取 拟合结果
w = lr.coef_  # 系数
b = lr.intercept_ # 截距
cost = compute_cost(w, b, points)
print('-------------------- 拟合结果 ----------------------------')
print(' w是: ', w)
print(' b是: ', b)
print(' 损失值: ', cost)


# 画出拟合曲线
w_num = w[0][0]
b_new = b[0]
plt.scatter(x, y)
## 针对每一个x, 计算出预测的y值
pred_y = w_num * x + b_new  # 因为x是一个向量, 所以计算出的结果也是一个向量(矩阵)
## 画出线
plt.plot(x, pred_y, c='r')
plt.show()

