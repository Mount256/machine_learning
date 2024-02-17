import numpy as np
import matplotlib.pyplot as plt

# 计算误差均方函数 J(w,b)
def cost_function(X, y, w, b):
    m = X.shape[0] # 训练集的数据样本数
    cost_sum = 0.0
    for i in range(m):
        f_wb_i = np.dot(w, X[i]) + b
        cost = (f_wb_i - y[i]) ** 2
        cost_sum += cost
    return cost_sum / (2 * m)

# 计算梯度值 dJ/dw, dJ/db
def compute_gradient(X, y, w, b):
    m = X.shape[0] # 训练集的数据样本数（矩阵行数）
    n = X.shape[1] # 每个数据样本的维度（矩阵列数）
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m): # 每个数据样本
        f_wb_i = np.dot(w, X[i]) + b
        for j in range(n): # 每个数据样本的维度
            dj_dw[j] += (f_wb_i - y[i]) * X[i, j]
        dj_db += (f_wb_i - y[i])
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

# 梯度下降算法
def linear_regression(X, y, w, b, learning_rate=0.01, epochs=1000):
    J_history = [] # 记录每次迭代产生的误差值
    for epoch in range(epochs):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        # w 和 b 需同步更新
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        J_history.append(cost_function(X, y, w, b)) # 记录每次迭代产生的误差值
    return w, b, J_history

# 绘制散点图
def draw_scatter(x, y, title):
    plt.xlabel("X-axis", size=15)
    plt.ylabel("Y-axis", size=15)
    plt.title(title, size=20)
    plt.scatter(x, y)

# 打印训练集数据和预测值数据以便对比
def print_contrast(train, prediction, n):
    print("train  prediction")
    for i in range(n):
        print(np.round(train[i], 4), np.round(prediction[i], 4))

# 从这里开始执行
if __name__ == '__main__':
    # 训练集样本
    data = np.loadtxt("./data.txt", delimiter=',', skiprows=1)
    X_train = data[:, :4] # 训练集的第 0-3 列为 X = (x0, x1, x2, x3)
    y_train = data[:, 4] # 训练集的第 4 列为 y
    w = np.zeros((X_train.shape[1],)) # 权重
    b = 0.0 # 偏置
    epochs = 1000  # 迭代次数
    learning_rate = 1e-7  # 学习率
    J_history = []  # 记录每次迭代产生的误差值

    # 线性回归模型的建立
    w, b, J_history = linear_regression(X_train, y_train, w, b, learning_rate, epochs)
    print(f"result: w = {np.round(w, 4)}, b = {b:0.4f}")  # 打印结果

    # 训练集 y_train 与预测值 y_hat 的对比（这里其实我偷了个懒，训练集当测试集用，以后不要这样做！）
    y_hat = np.zeros(X_train.shape[0])
    for i in range(X_train.shape[0]):
        y_hat[i] = np.dot(w, X_train[i]) + b
    print_contrast(y_train, y_hat, y_train.shape[0])

    # 绘制误差值的散点图
    x_axis = list(range(0, epochs))
    draw_scatter(x_axis, J_history, "Cost Function in Every Epoch")
    plt.show()
