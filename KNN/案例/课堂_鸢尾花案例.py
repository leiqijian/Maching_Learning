import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler                    #数据标准化
from sklearn.neighbors import KNeighborsClassifier                  #KNN算法，分类对象
from sklearn.metrics import accuracy_score                          #模型评估，计算模型预测的准确率


# 数据探索
def load_iris_data():

    iris_data = load_iris()

    # print(iris_data.data)
    print(iris_data.keys())
    print(iris_data.data)
    print(iris_data.target)
    print(iris_data.target_names)
    print(iris_data.feature_names)

    # 构建df对象
    iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)

    iris_df ["target"] = iris_data.target

    print(iris_df.head())

    # 查看相关性



    # 绘制图形
    sns.lmplot(x = 'petal length (cm)', y='petal width (cm)',hue='target',data=iris_df, fit_reg = False)
    plt.show()

def knn_iris():
    # 加载数据
    iris_data = load_iris()

    # 数据基本处理
    x_train, x_test, y_train, y_test = train_test_split(iris_data["data"], iris_data["target"], test_size=0.2, random_state=7, shuffle=True)
    # print(x_train)
    # print(x_test)
    # print(y_train)
    # print(y_test)

    # 特征预处理
    transformer = StandardScaler()
    transformer.fit(x_train)
    new_x_train = transformer.transform(x_train)
    new_x_test = transformer.transform(x_test)

    # 模型训练
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(new_x_train, y_train)

    # 模型评估
    y_pred = model.predict(new_x_test)
    print(accuracy_score(y_test, y_pred))

    # 模型上线







if __name__ == '__main__':
    # load_iris_data()

    knn_iris()
