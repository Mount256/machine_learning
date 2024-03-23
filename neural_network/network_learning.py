# coding: utf-8

import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

# ============================= activation functions ===================================

# sigmoid 函数
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# 改良的 softmax 函数（防止在指数运算时发生溢出）
def softmax_function(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

# ReLU 函数
def relu_function(x):
    return np.maximum(0, x)

# 线性激活函数（恒等函数）
def linear_activation_function(x):
    return x

# ============================= loss functions ===================================

# 均方误差函数
# 监督数据 t 为 one-hot 表示
def mean_squared_error(y, t):
    m = y.shape[1]
    cost_sum = np.sum((y-t) ** 2)
    return cost_sum / (2 * m)

# 对数损失函数
# 监督数据 t 为 one-hot 表示
def log_loss_function(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    size = y.shape[0]
    return -np.sum(np.log(y[np.arange(size), t] + 1e-7)) / size

# 数值微分（中心差分方法）
# f 为以上损失函数二选一
def _numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]

        # 计算 f(x+h)
        x[i] = float(tmp) + h
        fxh1 = f(x)

        # 计算 f(x-h)
        x[i] = float(tmp) - h
        fxh2 = f(x)

        # 计算中心差分
        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp

    return grad

# 数值微分（中心差分方法），由于输入形状不同，所以需要分别执行不同操作
def numerical_gradient(loss_f, X):
    if X.ndim == 1:  # 若输入为一维数组
        return _numerical_gradient(loss_f, X)
    else:  # 若输入为二维矩阵
        grads = np.zeros_like(X)
        for idx, x in enumerate(X): # enumerate函数：同时列出数据索引和数据
            grads[idx] = _numerical_gradient(loss_f, x)
        return grads

# ============================= neural network ===================================

class TwoLayerNet: # 一个有 2 层神经网络（输入层+隐藏层+输出层）的类

    # 初始化配置各个神经元的参数
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {} # 字典类型
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 高斯分布初始化，需注意矩阵的大小！
        self.params['B1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['B2'] = np.zeros(output_size)

    # 神经元的内部实现：输入A，权重W，偏置B，激活函数g()，输出A_out
    def dense(self, A, W, B, g):
        Z = np.matmul(A, W) + B # 这里是矩阵乘法，而非点乘
        A_out = g(Z)
        return A_out

    # 神经网络的搭建
    def predict(self, X):
        W1, W2 = self.params['W1'], self.params['W2']
        B1, B2 = self.params['B1'], self.params['B2']
        A1 = self.dense(X, W1, B1, sigmoid_function) # layer 1
        A2 = self.dense(A1, W2, B2, softmax_function) # layer 2
        return A2

    # 计算损失/误差值（回调函数）
    # loss_f 为损失函数
    def loss(self, loss_f, X, t):
        y = self.predict(X)  # 先进行推理
        return loss_f(y, t)  # 后用推理结果与正确解标签计算损失值

    # 梯度下降算法（采用数值微分方法）
    # loss_f 为损失函数
    def numerical_gradient_descent(self, loss_f, X, t):
        # 定义匿名函数，参数为 W，返回值为 loss_f(y, t)
        loss_W = lambda W: self.loss(loss_f, X, t)
        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['B1'] = numerical_gradient(loss_W, self.params['B1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['B2'] = numerical_gradient(loss_W, self.params['B2'])

        return grads

    # 神经网络的学习
    def learning(self, X, t, loss_f=mean_squared_error, learning_rate=0.01, iters=1000):
        self.loss_history = [] # 记录每次迭代的损失值
        self.grads = {}  # 记录参数 W 和 B 的梯度值

        for i in range(iters):
            # 梯度下降算法
            self.grads = self.numerical_gradient_descent(loss_f, X, t)
            for key in ('W1', 'B1', 'W2', 'B2'):
                self.params[key] -= learning_rate * self.grads[key]

            # 识别精度
            accuracy = self.accuracy(X, t, X.shape[0])
            print(f"iter {i}: train accuracy = {accuracy}")

            # 计算损失值
            loss = self.loss(loss_f, X, t)
            self.loss_history.append(loss)

        return self.params, self.loss_history

    # 神经网络参数的精度评价
    def accuracy(self, X, t, num):
        y = self.predict(X)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(num)
        return accuracy

# ============================= other ===================================

# 绘制散点图
def draw_scatter(x, y, title):
    plt.xlabel("X-axis", size=15)
    plt.ylabel("Y-axis", size=15)
    plt.title(title, size=20)
    plt.scatter(x, y)

# ============================= main ===================================

if __name__ == '__main__':
    # 获取训练集和测试集（一维化、归一化、采用one-hot标签）
    # x_train: (60000,784), t_train: (60000,10)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)
    learning_rate = 0.1  # 学习率
    iters = 1000  # 迭代次数
    loss_history = []
    params = {}

    # 初始化神经网络的参数，输入层 784 个神经元，隐藏层 50 个神经元，输出层 10 个神经元
    network = TwoLayerNet(input_size = x_train.shape[1],
                          hidden_size = 50,
                          output_size = t_train.shape[1])

    # 二层神经网络的学习（只学习前 100 个图像）
    params, loss_history = network.learning(X = x_train[0:100, :],
                                            t = t_train[0:100, :],
                                            loss_f = log_loss_function,
                                            learning_rate = learning_rate,
                                            iters = iters)

    # 绘制误差值的散点图
    x_axis = list(range(0, iters))
    draw_scatter(x_axis, network.loss_history, "Loss Function in Every Epoch")
    plt.show()
