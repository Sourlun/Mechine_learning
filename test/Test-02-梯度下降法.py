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


# 定义模型的超参数
alpha = 0.00001  # 步长
init_w = 0  # 可以随机选取, 然后试多几个看看损失函数是否最小
init_b = 0  # 可以随机选取, 然后试多几个看看损失函数是否最小
num_iter = 100  # 迭代次数


# 定义核心 梯度下降算法函数
def grad_desc(points, init_w, init_b, alpha, num_iter):
    w = init_w
    b = init_b
    # 定义一个列表, 保存所有损失函数的值, 用来显示下降过程
    cost_list = []

    # 根据迭代次数迭代
    for i in range(num_iter):
        cost_list.append(compute_cost(w, b, points))
        w, b = step_grad_desc(w, b, alpha, points)

    return [w, b, cost_list]

# 计算梯度
## 资料参考:梯度下降法求解线性回归-01.png
def step_grad_desc(current_w, current_b, alpha, points):
    sum_grad_w = 0
    sum_grad_b = 0
    m = len(points)

    # 对每个点, 带入公式求和
    for i in range(m):
        x = points[i, 0]
        y = points[i, 1]
        sum_grad_w += ((current_w * x) + current_b - y) * x
        sum_grad_b += (current_w * x) + current_b - y

    grad_w = 2 / m * sum_grad_w
    grad_b = 2 / m * sum_grad_b

    # 更新当前的w和b
    updated_w = current_w - alpha * grad_w
    updated_b = current_b - alpha * grad_b

    return updated_w, updated_b

# 测试
## 运行梯度下降算法计算最优的w和b
w, b, cast_list = grad_desc(points, init_w, init_b, alpha, num_iter)
print(' -----------------  拟合结果 START -----------------------')

print('w: ', w, ',  b: ', b)

## 损失结果
cost = compute_cost(w, b, points)
print('损失结果: ', cost)

print(' ===================  拟合结果 END ====================')

# 损失函数
plt.plot(cast_list)
plt.show()

# 拟合曲线
pred_y = w * x + b
plt.scatter(x, y)
plt.plot(x, pred_y, c='r')
plt.show()

