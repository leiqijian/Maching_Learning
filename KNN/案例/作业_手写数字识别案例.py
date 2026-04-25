import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 数据集划分
from sklearn.neighbors import KNeighborsClassifier # 本次是分类问题
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 了解：如果不想看这个警告UserWarning: X does not have valid feature names, but KNeighborsClassifier was fitted with feature names
#   warnings.warn(
# 可以添加如下代码
import warnings
warnings.filterwarnings(action="ignore", module="sklearn")

def analyse_num():
    # load data
    num_data = pd.read_csv("data/手写数字识别.csv")

    num_target = num_data["label"]
    num_feature = num_data.iloc[:, 1:]

    x_train, x_test, y_train, y_test = train_test_split(num_feature, num_target, test_size=0.2, random_state=22, shuffle=True)

    transformer = MinMaxScaler()
    new_x_train = transformer.fit_transform(x_train)
    new_x_test = transformer.transform(x_test)

    model = KNeighborsClassifier(n_neighbors=3)

    # param_dict = {
    #     "n_neighbors": [i for i in range(1, 50)],
    # }
    #
    # model = GridSearchCV(estimator=model, param_grid=param_dict, cv=5)

    model.fit(new_x_train, y_train)

    # print(model.best_params_) #{'n_neighbors': 3}
    # print(model.best_estimator_) # KNeighborsClassifier(n_neighbors=3)
    # print(model.best_score_) # 0.964375

    unknown_img_df = plt.imread("data/demo.png")

    unknown_img_df = unknown_img_df.reshape(1, 784)

    y_pred = model.predict(new_x_test)

    print(accuracy_score(y_test, y_pred))

    print(model.predict(unknown_img_df))

if __name__ == '__main__':
    analyse_num()