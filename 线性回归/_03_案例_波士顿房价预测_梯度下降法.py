'''
用梯度下降法 线性回归对象 完成 波士顿房价预测案例

回顾：
    线性回归算法 属于 有监督学习：有特征，有标签，且标签是连续的
    线性回归分类：
        一元线性回归： 1个特征列，1个标签列
        多元线性回归： 多个特征，1个标签列
    线性回归大白话解释：
        它是用线性公式来描述 特征 和标签之间的关系的，公式如下
        一元线性回归公式： y = w * x + b
        多元线性回归公式： y = w1 * x1 + w2 * x2 + w3 * x3 + ... + wn * xx + b
                         = w的转置 * x + b
    如何衡量线性回归模型的好坏？
        思路：
            预测值 和 真实值 之间的误差，误差越小，模型越好 ==>损失函数
        具体方案：
            1. 最小二乘
            2. 均方误差 （MSE）
            3. 均方根误差 (RMSE)
            4. 平均绝对误差 (MAE)

    如何让损失函数最小
        1. 梯度下降
        2. 正规方程
'''

# from sklearn.datasets import load_boston                # 数据
from sklearn.preprocessing import StandardScaler        # 特征处理
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor           # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差评估
from sklearn.linear_model import Ridge, RidgeCV
import pandas as pd
import numpy as np
from sklearn.utils._repr_html import estimator

# 1. 加载波士顿房价数据
# data = load_boston()
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
# 2. 数据特征处理 （数据切割）

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=22)

# 3. 特征工程（标准化，归一化）
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 模型训练 （使用梯度下降）
# fit_intercept = 是否计算截距
# learning_rate = 学习率模式，使用常量
# eta0 :学习率
estimator = SGDRegressor(fit_intercept=True , learning_rate='constant', eta0=0.01)
estimator.fit(x_train, y_train)


# 5. 模型预测
y_pre = estimator.predict(x_test)
print(f"预测结果为{y_pre}")

# 6. 模型评估 （误差函数）
print(f"均方误差：{mean_squared_error(y_test, y_pre)}")
print(f"均方根误差：{root_mean_squared_error(y_test, y_pre)}")
print(f"平均绝对误差：{mean_absolute_error(y_test, y_pre)}")