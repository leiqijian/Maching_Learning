'''
原理：
基于欧式距离（或者其他距离计算方式）计算 测试集 和每个训练集之间的距离，然后根据距离升序排序，找到最近的K个样本
基于K个样本投票，票数多的就作为最终预测结果 -> 分类问题
基于K个样本的平均值，最终作为预测结果 -> 回归问题

实现思路
1. 分类问题
      适用于：有特征，有标签，且标签是不连续的（离散的）
2. 回归问题
      适用于： 有特征，有标签，且标签是连续的

KNN算法，分类问题思路如下：
      1. 计算测试集和每个训练集样本之间的距离
      2. 基于距离进行升序排序
      3. 找到最近的K个样本
      4. K个样本进行投票
      5. 票数多的结果，作为最终的预测结果
代码实现思路：
      1. 导包
      2. 准备数据集（测试集和训练集）
      3. 创建（KNN）分类模型对象
      4. 模型训练
      5. 模型预测
'''

from sklearn.neighbors import KNeighborsClassifier

# x_train = [[0], [1], [2], [3]]
# y_train = [0 , 0 , 1 , 1]
# x_test = [[5]]

x_train = [[0], [1], [2], [3]]
# y_train = ["a" , "a" , "b" , "b"]         #['a']
# y_train = ["b" , "b" , "z" , "z"]         #['z']
# y_train = ["b" , "b" , "a" , "a"]           #['a']
# y_train = ["中" , "中" , "国" , "国"]       #['中']
y_train = ["王" , "王" , "玉玉玉玉" , "玉玉玉玉"]       #['玉玉玉玉']
# y_train = ["爱情" , "爱情" , "文艺" , "文艺"]   #['文艺']
x_test = [[5]]

estimator  = KNeighborsClassifier(n_neighbors=4)
estimator.fit(x_train, y_train)
y_pred = estimator.predict(x_test)
print(f"result is {y_pred}")