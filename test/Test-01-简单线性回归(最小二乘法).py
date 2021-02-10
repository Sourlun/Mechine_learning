# 简单线性回归(最小二乘法)

# NumPy支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
import numpy as np
# matplotlib 是 Python 的绘图库。
# 它可与 NumPy 一起使用，提供了一种有效的 MatLab 开源替代方案。 它也可以和图形工具包一起使用
import matplotlib.pyplot as plt

# 导入数据
points = np.genfromtxt('../data/training/data.csv', delimiter=',')  # delimiter:分隔符
## 变成二维数组输出
print(points)

# 获取x,y轴的数据
## [:全部,每一项数组的第几个]
x = points[:, 0]
y = points[:, 1]

# 用matplotlib.pyplot画出散点图 (下面有画, 这里不重复)
# plt.scatter(x, y)  # scatter:散点
# plt.show()


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


# 求 均值 的函数
def average(dataList):
    sum = 0
    num = len(dataList)
    for i in range(num):
        sum += dataList[i]
    return sum / num


# 核心算法: 拟合函数
## 公式参考资料:求解线性回归.png下面
def fit(points):
    m = len(points)
    # x轴的平均数
    x_average = average(points[:, 0])

    # 分子的累加和
    sum_up = 0
    # 分母左边的累加
    sum_down_left = 0
    # 分母右边的累加
    sum_down_right = 0

    # 赋值三个累加值
    for i in range(m):
        x = points[i, 0]
        y = points[i, 1]
        sum_up += y * (x - x_average)
        sum_down_left += x ** 2
        sum_down_right += x  # 还没算完
    sum_down_right = (sum_down_right ** 2) / m

    # 根据公式计算系数
    w = sum_up / (sum_down_left - sum_down_right)
    sum_delta = 0
    for i in range(m):
        x = points[i, 0]
        y = points[i, 1]
        sum_delta += y - (w * x)
    b = sum_delta / m

    return w, b


# 输出
w, b = fit(points)
print(' -----------------  拟合结果 START -----------------------')

print('w: ', w, ',  b: ', b)

## 损失结果
cost = compute_cost(w, b, points)
print('损失结果: ', cost)

print(' ===================  拟合结果 END ====================')


# 画出拟合曲线
plt.scatter(x, y)
## 针对每一个x, 计算出预测的y值
pred_y = w * x + b  # 因为x是一个向量, 所以计算出的结果也是一个向量(矩阵)
## 画出线
plt.plot(x, pred_y, c='r')
plt.show()


