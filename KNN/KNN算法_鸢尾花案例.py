'''
案例： 通过KNN算法实现鸢尾花案例的分类操作

回顾：机器学习项目的研发流程
    1. 加载数据
    2. 数据的预处理
    3. 特征工程（提取，预处理）
    4. 模型训练
    5. 模型评估
    6. 模型预测
'''

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def dm01_load_Iris():
    #1. 加载鸢尾花数据集
    iris_data = load_iris()
    #2. 查看数据集
    # print(iris_data)
    # #3. 查看数据集的所有键
    # print(iris_data.keys()) #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    #4. 查看数据集的键对应的值，提取一部分，做切割
    print(iris_data.data)











if __name__ == '__main__':
    dm01_load_Iris()