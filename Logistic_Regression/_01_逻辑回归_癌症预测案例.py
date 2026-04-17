'''
案例：癌症预测

逻辑回归模型介绍：
    概述：
        属于有监督学习，有特征，有标签，且标签是连续的
        主要使用于二分类问题。0~1之间概率
    原理：
        把线性回归处理后的预测值，通过 sigmoid激活函数 映射到【0.1】 之间的概率，最后基于自定义阈值，预测类型
    逻辑回归的损失函数：
        极大似然估计的负数形式

机器学习的基本建模流程
    1. 加载数据集
    2. 数据基本处理
    3. 特征预处理
    4. 模型训练
    5. 模型预测
    6. 模型评估
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def logistic_regression():
    # 1. 加载数据集
    data = pd.read_csv('breast-cancer-wisconsin.csv')
    print(type(data)) #DataFrame
    print(data.info())
    # 2. 数据基本处理
    data.replace(to_replace='?', value=np.nan, inplace=True)
    data.dropna(axis=0, inplace = True)

    # 3. 特征预处理
    # 3.1 数据提取
    x = data.iloc[:, 1:-1] #按照行列号，获取索引数据， ：表示所有行，1:-1表示第一列到最后一列
    y = data['Class'] #获取标签值
    print(x)
    print(y)
    # 3.2 数据切割
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=15)
    # 3.3 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. 模型训练
    LR = LogisticRegression()
    LR.fit(x_train, y_train)


    # 5. 模型预测
    y_pred = LR.predict(x_test)   
    print(y_pred)
    # 6. 模型评估
    # 正确率（准确率），公式： 预测对的 / 样本总数
    print(LR.score(x_test, y_test)) #预测前评估，测试集的特征，标签
    print(accuracy_score(y_test, y_pred)) #预测前评估，测试集的标签, 预测值

    #逻辑回归的结果并不精准，只是大概给一个概率。如果需要精准，就需要通过混淆矩阵进行评测，即精确率，召回率，F1值，ROC曲线，AUC值等

if __name__ == '__main__':
    logistic_regression()