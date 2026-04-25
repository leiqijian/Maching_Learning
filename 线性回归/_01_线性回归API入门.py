'''
线性回归介绍
    概述/目的：
        用线性公式 来描述 多个变量（特征） 和 1个因变量（标签）之间的 关系，对其进行建模，基于特征 预测 标签
        线性回归属于：有监督学习，即：有特征，有标签，且标签连续
    分类：
        一元线性回归： 1个特征 + 1个标签
        多元线性回归： 多个特征 + 1个标签
    公式：
        一元线性回归：
            y = kx + b => wx + b
                k: 数学中叫斜率，机器学习中叫权重，也叫weight，w
                b：数学中叫截距，机器学习中叫偏置。bias。b
        多元线性回归：
            y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5... + b
              = w的转置 * x + b
'''
from sklearn.linear_model import LinearRegression


def demo():
# 1 准备数据身高和体重
    x_train = [[160], [166],  [172], [174],[180]]
    y_train = [56.3, 60.6, 65.1, 68.5, 75]
    x_test = [[176]]

# 2 数据预处理
# 3 特征工程

# 4 训练模型
    estimator =  LinearRegression().fit(x_train, y_train)

    y_test = estimator.predict(x_test)

    print(y_test)
    print(estimator.coef_)
    print(estimator.intercept_)



# 5 模型预测

if __name__ == '__main__':
    demo()

