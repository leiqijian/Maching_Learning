import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler                    #数据标准化
from sklearn.neighbors import KNeighborsClassifier                  #KNN算法，分类对象
from sklearn.metrics import accuracy_score                          #模型评估，计算模型预测的准确率

def knn_iris():
    # 加载数据
    iris_data = load_iris()

    # 数据基本处理
    x_train, x_test, y_train, y_test = train_test_split(iris_data["data"], iris_data["target"], test_size=0.2, random_state=7, shuffle=True)

    # 特征预处理
    transformer = StandardScaler()
    transformer.fit(x_train)
    new_x_train = transformer.transform(x_train)
    new_x_test = transformer.transform(x_test)

    # 模型训练
    model = KNeighborsClassifier(8)

    param_grid = {
        'n_neighbors': [i for i in range(1, 10)],
    }
    # print(f"model: {model}")
    # model = GridSearchCV(estimator=model, param_grid=param_grid, cv=4)
    # GridSearchCV(estimator=model, param_grid=param_grid, cv=4)
    # print(f"model_before: {model}")

    model.fit(new_x_train, y_train)
    # print(f"model_after: {model}")

    # print(model.best_params_)
    # print(model.best_score_)
    # print(model.best_params_)


    # 模型评估
    y_pred = model.predict(new_x_test)
    print(accuracy_score(y_test, y_pred))




if __name__ == '__main__':
    # load_iris_data()
    knn_iris()
