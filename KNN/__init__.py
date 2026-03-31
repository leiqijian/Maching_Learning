import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# ---------- 1. 构建数据集（使用前8行完整数据）----------
data = {
    '搞笑镜头': [39, 3, 2, 9, 8, 5, 21, 45],
    '拥抱镜头': [0, 2, 3, 38, 34, 2, 17, 2],
    '打斗镜头': [31, 65, 55, 2, 17, 57, 5, 9],
    '电影类型': ['喜剧片', '动作片', '爱情片', '爱情片', '爱情片', '动作片', '喜剧片', '喜剧片'],
    '距离': [21.47, 52.01, 43.42, 40.57, 34.44, 43.87, 18.55, 23.43]   # 假设这是需要回归的目标值
}

df = pd.DataFrame(data)
print("原始数据：")
print(df)

# ---------- 2. 特征和目标变量 ----------
X = df[['搞笑镜头', '拥抱镜头', '打斗镜头']]   # 特征

# 分类任务：预测电影类型
y_class = df['电影类型']

# 回归任务：预测距离（连续值）
y_reg = df['距离']

# ---------- 3. 划分训练集和测试集 ----------
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.3, random_state=42, stratify=y_class
)
_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

# ---------- 4. 特征标准化（KNN必须做）----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 5. KNN 分类 ==========
print("\n" + "="*40)
print("KNN 分类任务（预测电影类型）")
print("="*40)

knn_clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_clf.fit(X_train_scaled, y_train_class)
y_pred_class = knn_clf.predict(X_test_scaled)

print("测试集真实类型：", list(y_test_class))
print("预测类型：       ", list(y_pred_class))
print(f"分类准确率：{accuracy_score(y_test_class, y_pred_class):.2f}")

# ========== 6. KNN 回归 ==========
print("\n" + "="*40)
print("KNN 回归任务（预测距离）")
print("="*40)

knn_reg = KNeighborsRegressor(n_neighbors=3, metric='euclidean')
knn_reg.fit(X_train_scaled, y_train_reg)
y_pred_reg = knn_reg.predict(X_test_scaled)

print("测试集真实距离：", list(y_test_reg))
print("预测距离：       ", [round(x, 2) for x in y_pred_reg])
print(f"均方误差（MSE）：{mean_squared_error(y_test_reg, y_pred_reg):.2f}")
print(f"R² 分数：        {r2_score(y_test_reg, y_pred_reg):.2f}")

# ---------- 可选：对新样本进行预测 ----------
new_sample = [[20, 10, 8]]   # 假设一个新电影：搞笑20，拥抱10，打斗8
new_sample_scaled = scaler.transform(new_sample)

print("\n" + "="*40)
print("对新样本的预测")
print("="*40)
print(f"新样本特征：{new_sample[0]}")
print(f"预测电影类型：{knn_clf.predict(new_sample_scaled)[0]}")
print(f"预测距离：{knn_reg.predict(new_sample_scaled)[0]:.2f}")