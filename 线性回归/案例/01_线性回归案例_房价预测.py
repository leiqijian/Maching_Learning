import pandas as pd
from sklearn.model_selection import train_test_split    # 训练集和测试集划分
from sklearn.preprocessing import StandardScaler    # 标准化处理
from sklearn.linear_model import LinearRegression   # 普通线性回归
from sklearn.metrics import mean_squared_error  # MSE均方误差
from sklearn.metrics import mean_absolute_error # MAE平均绝对误差
from sklearn.metrics import root_mean_squared_error # RMSE均方根误差
from sklearn.linear_model import SGDRegressor           # 梯度下降的回归模型
from sklearn.model_selection import GridSearchCV

def ml():
    return LinearRegression()

def sgd_ml():
    return SGDRegressor(fit_intercept=True , learning_rate='constant', eta0=0.01)

def sgd_gridcv_ml():
    model = SGDRegressor()
    '''
        max_iter: 
        learning_rate: 
            constant: 
        eta0:
        penalty:
        alpha:
        
    '''
    param_grid = {
        'max_iter': [500 + i * 100 for i in range(50)],
        'learning_rate' : ['constant', 'invscaling', 'adaptive'],
        'eta0': [0.001 + i  for i in range(100)],
    }

    GridSearchModel = GridSearchCV(estimator=model, param_grid=param_grid, cv=4)

    return GridSearchModel


def train(model):
    house_price_data = pd.read_excel('data/boston_house_prices.xlsx')

    x = house_price_data.iloc[:, :-1]
    y = house_price_data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 22, shuffle = True)

    transformer = StandardScaler()
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse =mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    print('MSE:',mse)
    print('MAE:',mae)
    print('RMSE:',rmse)


if __name__ == '__main__':
    # 普通的线性回归
    print("普通的线性回归")
    ml_model = ml()
    train(ml_model)

    # 随机梯度下降
    print("随机梯度下降")
    sgd_model = sgd_ml()
    train(sgd_model)

    # 随机梯度下降+交叉验证和网格搜索
    print("随机梯度下降+交叉验证和网格搜索")
    sgd_gridcv_model = sgd_gridcv_ml()
    train(sgd_gridcv_model)