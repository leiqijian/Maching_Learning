import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier # 决策树：分类
from sklearn.metrics import precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree  # 用来绘制决策树


def train():
    # load data
    df = pd.read_csv("data/train.csv")
    # print(df.head())
    # print(df.info())

    # handle data
    # inplace ?
    df["Age"] = df["Age"].fillna(value=df["Age"].median())
    drop_df = df.drop(columns=["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"])

    #one hot
    one_hot_df = pd.get_dummies(drop_df)
    # print(one_hot_df.info())

    x = one_hot_df[["Pclass", "Age", "Sex_female"]]
    y = one_hot_df.iloc[:, 0]
    # print(x.head())
    # print(y.head())

    # split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=505, stratify=y, shuffle=True)

    transformer = StandardScaler()
    transformer.fit(x_train)
    x_train = transformer.transform(x_train)
    x_test = transformer.transform(x_test)

    """
        参数解释：
            criterion：具体使用什么样的分类决策树。有如下的取值
                gini：使用CART分类决策树
                entropy：使用ID3或者C4.5分类决策树，程序底层会自动根据你的数据和代码来决定用什么树

            max_depth：决策树的最大深度。经常设置该参数
            min_samples_split：子节点中样本条数的下限
            min_samples_leaf：叶节点（末端）中样本条数的下限
            ccp_alpha：成本计算函数
    """
    model = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=2, min_samples_leaf=1,
                                   ccp_alpha=0.0)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("精确率：",precision_score(y_test, y_pred))
    print("召回率：",recall_score(y_test, y_pred))
    print("F1值：",f1_score(y_test, y_pred))

    plt.figure(figsize=(20,20),dpi=300)
    """
        参数解释：
            decision_tree：决策树模型实例对象
            max_depth：展示出来的决策树的最大深度
            feature_names：展示的特征字段名称。注意：必须与上面x中特征的字段名称顺序一致
            class_names：展示的目标值对应的名称，可以任意
            filled：是否在决策树的格子中展示信息
    """
    plot_tree(
        decision_tree=model,
        max_depth=5,
        feature_names=["Pclass", "Age", "Sex_female"],
        class_names=["No_Survived","Yes_Survived"],
        filled=True
    )
    plt.savefig("data/泰坦尼克号案例.jpg")
    plt.show()

if __name__ == '__main__':
    train()
