from sklearn.datasets import make_blobs  # 产生随机的测试数据
from sklearn.cluster import KMeans  # KMeans无监督聚类算法
from sklearn.metrics import calinski_harabasz_score  # CH轮廓系数
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 1- 准备数据
    """
        参数解释：
            n_samples：样本条数
            n_features：每个样本的特征个数。n_features=2，就是有2个特征，那么其中一个特征用来作为横轴，另一个特征用来作为纵轴
            centers：质心坐标点
            cluster_std：簇的标准差。该值越大，样本数据越离散

        返回值解释：
            x：存储所有随机产生的特征列的值，目前有2个特征
            y：目标值。别人帮你产生随机的样本数据，它是知道每条样本数据的真实目标值是什么样的
    """
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.2, 0.2],
        random_state=430
    )
    # print(x)
    # print(y)

    # 2- 模型训练
    # 2.1- 创建模型实例对象
    model = KMeans(n_clusters=4, max_iter=2000)

    # 2.2- 模型训练和预测
    model.fit(x)

    y_pred = model.predict(x)


    # 3- 绘制散点图
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    # plt.show()

    # 4- 聚类评估指标
    """
        CH轮廓系数要求n_clusters的超参数值至少从2开始，不能从1开始。否则报错：
        ValueError: Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)
    """
    print("CH轮廓系数：", calinski_harabasz_score(x, model.labels_))
    print(f"y_pred : {y_pred}")
    print(f"model.labels_ : {model.labels_}")
