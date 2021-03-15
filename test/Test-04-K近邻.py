import numpy as np
# 科学计算, 数值分析
import pandas as pd

# 直接引入sklearn里面的数据集, iris某个花的数据
from sklearn.datasets import load_iris
# 切分数据集为训练集合测试集合
from sklearn.model_selection import train_test_split
# 用来计算分类预测的准确率
from sklearn.metrics import accuracy_score

# 花的数据
iris = load_iris()
print(type(iris))
print(iris)

print()
print()

# 可视化数据
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['class'] = iris.target  # 添加多一列, 用于查看结果
df['class_name'] = df['class'].map(
    {0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})  # 把iris.target对应成文字输出

print(df)

print()
print()

# 把平均, 方差, 最小, 输出, 中位数, 中位数75%...
print(df.describe())

# 拿数据
x = iris.data
y = iris.target.reshape(-1, 1)  # 任意行, 一列的矩阵
print('x的形状:', x.shape, '  y的形状:', y.shape)
# print(y)


# 划分 训练集 和 测试集   这一步仅仅只是划分
# train_test_split()函数是用来随机划分样本数据为训练集和测试集的，当然也可以人为的切片划分。
# train_test_split(train_data,train_target,test_size=0.3,random_state=5)
# train_data：待划分样本数据
# train_target：待划分样本数据的结果（标签）
# test_size：测试数据占样本数据的比例，若整数则样本数量
# random_state：设置随机数种子，保证每次都是同一个随机数。若为0或不填，则每次得到数据都不一样
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=30, stratify=y)
print('-------- x的训练集 --------')
print(x_train.shape)
# print(x_train)
print('-------- x的测试集 --------')
print(x_test.shape)
# print(x_test)
print('-------- y的训练集 --------')
print(y_train.shape)
# print(y_train)
print('-------- y的测试集 --------')
print(y_test.shape)


# print(y_test)


#
#  --------------- 核心算法 --------------------
#

#
# 距离函数定义 1 (曼哈顿距离)
#   a: 矩阵
#   b: 一行y列向量
#   sqrt( sum(abs(a-b)) )
#   if: a=x行y列 -> 那么b=1行y列
#
def l1_distance(a, b):
    abs = np.abs(a - b)  # a:矩阵, b:向量  np规定的
    sum = np.sum(abs, axis=1)  # axis=1: 保存成一列,  不加axis -> 返回a的行和a的列
    sqrt = np.sqrt(sum)
    return sqrt


#
# 距离函数定义 2 (欧氏距离)
#   a: 矩阵
#   b: 一行y列向量
#   sqrt( sum((a-b)**2) )
#   if: a=x行y列 -> 那么b=1行y列
#
def l2_distance(a, b):
    sum = np.sum( ((a - b) ** 2), axis=1) # sum: 1行n列列表  (把每一个变量的距离都加起来)
    sqrt = np.sqrt(sum)  # sqrt: 1行n列列表
    return sqrt


#
# 分类器实现
#
class Knn(object):
    #
    # 初始化方法, 构造方法
    #   n_neighbors: 相当于k
    #
    def __init__(self, n_neighbors=1, dist_func=l1_distance):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

    #
    # 训练模型的方法
    #   就是为了赋值, 其他啥都不做
    #
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    #
    # 模型的预测方法
    #   x: 需要预测的点, 这里是x_test
    #
    def predict(self, x):
        # 初始化预测数组  ( 传入x多少行, 返回y就有多少行1列 )
        # numpy.zeros()的作用：通常是把数组转换成想要的矩阵；
        # 用法：zeros(shape, dtype=float, order='C')
        #   shape:数据尺寸  例如：zeros(5) ----就是包含5个元素的零矩阵，默认dtype=float
        # （没有填充数据，默认为0的矩阵---零矩阵）
        # x.shape[0]: 获取x的行数
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)

        # 遍历输入的x
        # enumerate: 返回一个 (key, value)的元组
        for i, x_test in enumerate(x):
            #
            # 查看资料: KNN算法步骤.png
            #

            # 1, 跟所有训练数据计算距离
            distance = self.dist_func(self.x_train, x_test)  # 返回的是个x行的矩阵

            # 2, 排序计算完的矩阵distance, 取出索引
            nn_index = np.argsort(distance)  # 从小到大排序, 返回相应的索引列表

            # 3, 选取最近的k个点, 保存它们对应的分类类别
            nn_y = self.y_train[nn_index[:self.n_neighbors]]  # list[ :end ]: 获取list从头开始到end的元素, 返回列表
            nn_y = nn_y.ravel()  # ravel(): 将多维数组转换为一维数组, 这里是为了方便统计每个y出现的频率

            # 4, 统计类别中出现频率出现最高的那个, 赋值给 y_pred
            y_count = np.bincount(nn_y)  # bincount(): 统计列表元素出现次数  如: [2, 1, 0, 1, 1, 2] -> [1, 3, 2]
            y_pred[i] = np.nanargmax(y_count)  # nanargmax: 返回最大那个元素的索引   如: [1, 3, 2] -> 1

        return y_pred


print()
print('---------------------- 测试 ----------------------------')
print()

# 实例
knn = Knn(n_neighbors=5, dist_func=l1_distance)
# 训练
knn.fit(x_train, y_train)
# 传入测试数据, 做预测
y_pred = knn.predict(x_test)

# 求出预测准确率
#   accuracy_score(): 分类准确率分数是指所有分类正确的百分比
#       normalize：默认值为True，返回正确分类的比例；如果为False，返回正确分类的样本数
#       如: y_pred = [0, 2, 1, 3]; y_true = [0, 1, 2, 3] & normalize=True -> 0.5
#                                                        & normalize=False -> 2
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred, normalize=True)
print('预测准确率:', accuracy)


print()
print('---------------------- 测试 二 ----------------------------')
print()

# 实例
knn1 = Knn(n_neighbors=5, dist_func=l1_distance)
# 训练
knn1.fit(x_train, y_train)

# 结果list
results = []

for p in [1, 2]:
    # p == 1: 曼哈顿距离;  p == 2: 欧氏距离
    knn1.dist_func = l1_distance if p == 1 else l2_distance
    for k in range(1, 10, 2):
        knn1.n_neighbors = k
        y_pred = knn1.predict(x_test)
        # 准确率
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred, normalize=True)
        results.append([k, '曼哈顿距离' if p == 1 else '欧氏距离', accuracy])

df = pd.DataFrame(results, columns=['k(邻居数量)', '距离函数', '预测准确率'])
print(df)




exit(-1)
