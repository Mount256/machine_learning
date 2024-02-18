import numpy as np
import matplotlib.pyplot as plt

# sigmoid 函数 f = 1/(1+e^(-x))
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

# 计算分数 z = w*x+b
def score(x, w, b):
    return np.dot(w, x) + b

# 预测值 f_pred = sigmoid(z)
def prediction(x, w, b):
    return sigmoid(score(x, w, b))

# 对数损失函数 f = -y*ln(a)-(1-y)*ln(1-a)
# 训练样本: (vec{X[i]}, y[i])
def log_loss(X_i, y_i, w, b):
    pred = prediction(X_i, w, b)
    return - y_i * np.log(pred) - (1-y_i) * np.log(1-pred)

# 计算损失函数 J(w, b)
# 训练样本: (vec{X[i]}, y[i])
def cost_function(X, y, w, b):
    cost_sum = 0
    m = X.shape[0]
    for i in range(m):
        cost_sum += log_loss(X[i], y[i], w, b)
    return cost_sum / m

# 计算梯度值 dJ/dw, dJ/db
def compute_gradient(X, y, w, b):
    m = X.shape[0]  # 训练集的数据样本数（矩阵行数）
    n = X.shape[1]  # 每个数据样本的维度（矩阵列数，即特征个数）
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m):  # 每个数据样本
        pred = prediction(X[i], w, b)
        for j in range(n):  # 每个数据样本的维度
            dj_dw[j] += (pred - y[i]) * X[i, j]
        dj_db += (pred - y[i])
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

# 梯度下降算法，以得到决策边界（decision boundary）方程
def logistic_function(X, y, w, b, learning_rate=0.01, epochs=1000):
    J_history = []
    for epoch in range(epochs):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        # w 和 b 需同步更新
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        J_history.append(cost_function(X, y, w, b))  # 记录每次迭代产生的误差值
    return w, b, J_history

# 绘制线性方程的图像
def draw_line(w, b, xmin, xmax, title):
    x = np.linspace(xmin, xmax)
    y = w * x + b
    plt.xlabel("feature-0", size=15)
    plt.ylabel("feature-1", size=15)
    plt.title(title, size=20)
    plt.plot(x, y)

# 绘制散点图
def draw_scatter(x, y, title):
    plt.xlabel("epoch", size=15)
    plt.ylabel("error", size=15)
    plt.title(title, size=20)
    plt.scatter(x, y)

# 从这里开始执行
if __name__ == '__main__':
    # 加载训练集
    X_train = np.array([[1, 0], [0, 2], [1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 2]])
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    w = np.zeros((X_train.shape[1],)) # 权重
    b = 0.0 # 偏置
    learning_rate = 0.01 # 学习率
    epochs = 10000 # 迭代次数
    J_history = [] # 记录每次迭代产生的误差值

    # 逻辑回归建立模型
    w, b, J_history = logistic_function(X_train, y_train, w, b, learning_rate, epochs)
    print(f"result: w = {np.round(w, 4)}, b = {b:0.4f}")  # 打印结果

    # 绘制迭代计算得到的决策边界（decision boundary）方程
    # w[0] * x_feature0 + w[1] * x_feature1 + b = 0
    # --> x_feature1 = -w[0]/w[1] * x_feature0 - b/w[1]
    plt.figure(1)
    draw_line(-w[0]/w[1], -b/w[1], 0.0, 3.0, "Decision Boundary")
    plt.scatter(X_train[0:4, 0], X_train[0:4, 1], label="label-0: sad", marker='s')  # 将训练集也表示在图中
    plt.scatter(X_train[4:8, 0], X_train[4:8, 1], label="label-1: happy", marker='^')  # 将训练集也表示在图中
    plt.legend()
    plt.show()

    # 绘制误差值的散点图
    plt.figure(2)
    x_axis = list(range(0, epochs))
    draw_scatter(x_axis, J_history, "Cost Function in Every Epoch")
    plt.show()
