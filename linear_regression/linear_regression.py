import numpy as np
import matplotlib.pyplot as plt

# 计算误差均方函数 J(w,b)
def cost_function(x, y, w, b):
    m = x.shape[0] # 训练集的数据样本数
    cost_sum = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    return cost_sum / (2 * m)

# 计算梯度值 dJ/dw, dJ/db
def compute_gradient(x, y, w, b):
    m = x.shape[0] # 训练集的数据样本数
    d_w = 0.0
    d_b = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        d_wi = (f_wb - y[i]) * x[i]
        d_bi = (f_wb - y[i])
        d_w += d_wi
        d_b += d_bi
    dj_dw = d_w / m
    dj_db = d_b / m
    return dj_dw, dj_db

# 梯度下降算法
def linear_regression(x, y, w, b, learning_rate=0.01, epochs=1000):
    J_history = [] # 记录每次迭代产生的误差值
    for epoch in range(epochs):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        # w 和 b 需同步更新
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        J_history.append(cost_function(x, y, w, b)) # 记录每次迭代产生的误差值
    return w, b, J_history

# 绘制线性方程的图像
def draw_line(w, b, xmin, xmax, title):
    x = np.linspace(xmin, xmax)
    y = w * x + b
    # plt.axis([0, 10, 0, 50]) # xmin, xmax, ymin, ymax
    plt.xlabel("X-axis", size=15)
    plt.ylabel("Y-axis", size=15)
    plt.title(title, size=20)
    plt.plot(x, y)

# 绘制散点图
def draw_scatter(x, y, title):
    plt.xlabel("X-axis", size=15)
    plt.ylabel("Y-axis", size=15)
    plt.title(title, size=20)
    plt.scatter(x, y)

# 从这里开始执行
if __name__ == '__main__':
    # 训练集样本
    x_train = np.array([1, 2, 3, 5, 6, 7])
    y_train = np.array([15.5, 19.7, 24.4, 35.6, 40.7, 44.8])
    w = 0.0 # 权重
    b = 0.0 # 偏置
    epochs = 10000 # 迭代次数
    learning_rate = 0.01 # 学习率
    J_history = [] # 记录每次迭代产生的误差值

    w, b, J_history = linear_regression(x_train, y_train, w, b, learning_rate, epochs)
    print(f"result: w = {w:0.4f}, b = {b:0.4f}") # 打印结果

    # 绘制迭代计算得到的线性回归方程
    plt.figure(1)
    draw_line(w, b, 0, 10, "Linear Regression")
    plt.scatter(x_train, y_train) # 将训练数据集也表示在图中
    plt.show()

    # 绘制误差值的散点图
    plt.figure(2)
    x_axis = list(range(0, epochs))
    draw_scatter(x_axis, J_history, "Cost Function in Every Epoch")
    plt.show()
