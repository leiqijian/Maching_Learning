import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 所有的分类算法都可以使用如下的评估指标
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score   # AUC面积
from sklearn.metrics import classification_report   # 分类评估报告

from collections import Counter # 类别计数

def logic_ml():
    # 1- 准备数据
    df = pd.read_csv("data/churn.csv")
    # print(df.describe())
    # print(df.info())

    # 2- 数据基本处理
    # 2.1- 得到特征和目标值
    x = df.iloc[:, 2:]
    y = df.iloc[:, 0]
    print(Counter(y))   # Counter({'No': 5174, 'Yes': 1869})

    # 2.2- 划分训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=429,shuffle=True,stratify=y)

    # 3- 特征工程
    transformer = StandardScaler()
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    # 4- 模型训练
    model = LogisticRegression(max_iter=8000)
    model.fit(x_train,y_train)

    # 5- 模型评估
    # 5.1- 进行预测
    y_pred = model.predict(x_test)

    # 5.2- 评估指标：精确率、召回率、F1值
    # 默认使用样本数据量少的作为正例，目前该案例中 目标值=Yes的样本数据少，是正例
    # pos_label：正例目标值内容
    print("精确率：",precision_score(y_test, y_pred, pos_label="Yes"))
    print("召回率：",recall_score(y_test, y_pred, pos_label="Yes"))
    print("F1值：",f1_score(y_test, y_pred, pos_label="Yes"))

    # 5.3- AUC面积
    # 获得预测概率值
    """
        predict_proba返回结果解释：
            1- 每个子列表代表的是每条样本的预测结果
            2- 有2列，第1列是反例的预测概率，第2列是正例的预测概率
            3- y_score=y_pred_proba[:,1]，y_score正例的预测概率值
    """
    y_pred_proba = model.predict_proba(x_test)
    # print(type(y_pred_proba))   # <class 'numpy.ndarray'>
    # print(y_pred_proba)
    print("AUC面积：",roc_auc_score(y_test,y_score=y_pred_proba[:,1]))

    # 5.4- 分类评估报告
    print("="*30)
    """
        分类评估报告指标解释：
            macro avg：宏平均=(正例概率+反例概率)/2
            weighted avg：加权平均=(正例概率*正例样本条数+反例概率*反例样本条数)/(正例样本条数+反例样本条数)
    """
    # digits：保留的小数位数
    print(classification_report(y_test, y_pred, digits=5))

def ong_hot_logic_ml():
    df = pd.read_csv("data/churn.csv")

    # 2- 数据基本处理
    # 2.1- one-hot独热编码处理
    """
        one-hot独热编码处理的目的：
            1- 算法模型无法直接处理非数值的特征值
            2- one-hot独热编码将非数值的特征值处理成0和1组成的结果。举例如下：
                A   [1,0,0]
                B   [0,1,0]
                C   [0,0,1]
    """
    one_hot_df = pd.get_dummies(df)

    drop_df = one_hot_df.drop(columns=["gender_Male", "Churn_No"], axis=0)

    x = drop_df.drop(columns=["Churn_Yes"], axis=0)
    y = drop_df.iloc[:, -2]



if __name__ == '__main__':
    # 普通版的逻辑回归
    logic_ml()