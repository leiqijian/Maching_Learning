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
from sklearn.model_selection import train_test_split,GridSearchCV   #分割训练集和测试架
from sklearn.preprocessing import StandardScaler                    #数据标准化
from sklearn.neighbors import KNeighborsClassifier                  #KNN算法，分类对象
from sklearn.metrics import accuracy_score                          #模型评估，计算模型预测的准确率

#加载数据
def dm01_load_Iris():
    #1. 加载鸢尾花数据集
    iris_data = load_iris()
    #2. 查看数据集
    # print(iris_data)
    # #3. 查看数据集的所有键
    # print(iris_data.keys()) #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    #4. 查看数据集的键对应的值，提取一部分，做切割
    print(iris_data.data)    #特征数据
    print(iris_data.target)  #具体的目标数据
    print(iris_data.target_names)  #标签对应的名称
    print(iris_data.feature_names) #特征对应的名称

#显示数据
def demo02_show():
    #加载数据
    myIrisData = load_iris()
    print(myIrisData.data)

    #把数据转换成dataframe格式，设置data，colums属性
    iris_d = pd.DataFrame(myIrisData.data, columns=myIrisData.feature_names)
    iris_d['label'] = myIrisData.target
    print(iris_d)

    # 计算特征字段与目标值之间的相关性： 皮尔逊系数，1为最大，越接近1相关性越高
    # 目的是方便从画图中看出哪些特征 ？？
    # 1.方式一 df.corr()

    # 2.方式二 通过scipy的pearsonr计算


    # 使用sns.lmplot()显示散点图
    # sns.lmplot(x='petal width (cm)',y='sepal length (cm)',data=iris_d,hue='label',fit_reg=False)

    sns.lmplot(y='petal width (cm)',x='sepal length (cm)',data=iris_d,hue='label',fit_reg=False)
    plt.show()

#数据集划分
def demo03_traintest_split():
    #1. 加载数据集
    iris_data = load_iris()
    # print(type(iris_data.data)) #<class 'numpy.ndarray'>
    # print(type(iris_data))  #<class 'sklearn.utils._bunch.Bunch'>
    # print(iris_data)

    #2. 使用train_test_split划分数据集,8:2划分
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=7)
    result = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=7)
    print(f"返回值的数量: {len(result)}")
    print(f"每个返回值的类型: {[type(r) for r in result]}")
    print(f"每个返回值的形状: {[r.shape for r in result]}")

def demo04():

    #1. 获取数据
    iris_data = load_iris()

    #2. 数据基本处理，划分数据
    # train_test_split(X, Y, A, B) X:特征值集合  Y:目标值集合  A:切分比例 B：随机数
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=22)


    #3. 特征工程（提取，预处理）
    # 特征提取：因为数据源只有4个特征列，且都是需要使用的，所以该案例无需做特征提取
    # 特征预处理：防止因为量纲（单位）的问题，导致特征列的方差值相差较大，影响模型的最终结果
    # 数据集预处理 （标准化 / 归一化）把训练集和测试集的特征数据标准化为0~1之间的数值
    # fit_transform: 兼具fit和transform的功能，即：训练, 转换。该函数适用于第一次标准化的时候使用。因为在创建了标准化对象StandardScaler()后
    # 该对象里自带一些默认参数，需要先把训练集的数据喂给该对象，使得参数调整到适应这个案例，后续就直接transform
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    # transform: 只有转换，适用于重复进行数据标准化
    x_test = transfer.transform(x_test)


    #4. 机器学习--模型训练
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(x_train, y_train)


    #5. 模型预测,喂测试特征值，得到测试标签值
    # 场景一：使用150条数据划分的测试集特征数据进行预测
    print("--------模型预测--------")
    y_prd = KNN.predict(x_test)
    print(y_prd)

    # 场景二： 自定义数据集进行预测
    # 自定义数据
    my_data = [[7.8 , 2.1, 3.9, 1.6]]
    # 对数据进行标准化操作
    x_test_new = transfer.transform(my_data)
    y_prd_new = KNN.predict(x_test_new)
    print(y_prd_new)

    # 查看上述数据集，每种分类的预测概率
    y_pre_proba = KNN.predict_proba(x_test_new)
    print(y_pre_proba)

    #6. 模型评估，评估模型准确性
    #方式一 ： 直接评分，基于 训练集特征值 和 训练集标签值 进行分析
    print("-------模型评估------")
    print(KNN.score(x_train, y_train))
    #方式二 : 基于 测试集的标签 （样本数据） 和 预测结果 （通过测试集特征值得到的预测结果） 进行评分
    print(accuracy_score(y_test, KNN.predict(x_test)))









if __name__ == '__main__':
    # dm01_load_Iris()
    # demo02_show()
    # demo03_traintest_split()
    demo04()