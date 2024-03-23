# coding: utf-8

import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from collections import OrderedDict

# ============================= 损失函数 loss functions ===================================

# 均方误差函数
# 监督数据 t 为 one-hot 表示
def mean_squared_error(y, t):
    m = y.shape[1]
    cost_sum = np.sum((y-t) ** 2)
    return cost_sum / (2 * m)

# 对数损失函数（交叉熵误差）
# 监督数据 t 为 one-hot 表示
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# ============================= 激活函数 activation functions ===================================
# =============== ReLU 函数 ===============
class ReLU:
    def __init__(self):
        self.mask = None

    # 前向传播
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    # 反向传播
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# =============== sigmoid 函数 ===============
class Sigmoid:
    def __init__(self):
        self.out = None

    # 前向传播
    def foward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    # 反向传播
    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx

# =============== Affine (Z = WX + B) ===============
class Affine:
    def __init__(self, W, B):
        self.W = W
        self.B = B
        self.X = None
        self.dW = None
        self.dB = None

    # 前向传播
    def forward(self, X):
        self.X = X
        out = np.matmul(self.X, self.W) + self.B
        return out

    # 反向传播
    def backward(self, dout):
        dX = np.matmul(dout, self.W.T)
        self.dW = np.matmul(self.X.T, dout)
        self.dB = np.sum(dout, axis=0)
        return dX

# =============== softmax with loss ===============
class SoftmaxWithLoss:
    def __init__(self, loss_f):
        self.y = None
        self.t = None
        self.loss = None
        self.loss_f = loss_f

    # 改良的 softmax 函数（防止在指数运算时发生溢出）
    def softmax_function(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)  # 溢出对策
        return np.exp(x) / np.sum(np.exp(x))

    # 前向传播
    def forward(self, x, t):
        self.y = self.softmax_function(x)
        self.t = t
        self.loss = self.loss_f(self.y, self.t)
        return self.loss

    # 反向传播
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

# ============================= 神经网络 neural network ===================================
class TwoLayerNet: # 一个有 2 层神经网络（输入层Affine + 隐藏层ReLU + 输出层Affine）的类

    # 初始化配置各个神经元的参数
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, loss_f=cross_entropy_error):
        self.params = {}  # 初始化权重，字典类型
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 高斯分布初始化，需注意矩阵的大小！
        self.params['B1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['B2'] = np.zeros(output_size)

        self.layers = OrderedDict() # 生成各种层，有序字典类型
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['B1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['B2'])
        self.lastLayer = SoftmaxWithLoss(loss_f)

    # 神经网络的前向传播（不包含 softmax 分类）
    def predict(self, X):
        for layer in self.layers.values(): # values()：获取字典所有的值，返回一个列表
            X = layer.forward(X)
        return X

    # 计算损失/误差值
    def loss(self, X, t):
        y = self.predict(X)  # 先进行推理（前向传播）
        return self.lastLayer.forward(y, t)  # 后用推理结果与正确解标签计算损失值

    # 梯度下降算法（采用误差反向传播法）
    def gradient_descent(self, X, t):
        # step1.前向传播
        self.loss(X, t)

        # step2.误差反向传播
        # 最后一层开始反向传播
        dout = 1
        dout = self.lastLayer.backward(dout)
        # 从后往前依次反向传播
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['B1'] = self.layers['Affine1'].dB
        grads['W2'] = self.layers['Affine2'].dW
        grads['B2'] = self.layers['Affine2'].dB

        return grads

    '''
    神经网络的学习（mini-batch 学习），总共学习 iters（1000）次，
    一共 train_size（60000）个数据，每次（iter）只随机看 batch_size（100）个数据，
    定义 train_size / batch_size = 一个 epoch（这里是 600 次 iter），
    因为 一个 epoch（600 iter/per epoch）* batch_size（100） = train_size（60000），
    所以每经过一轮 epoch（600 次 iter），就代表所有数据都被“查看过了”.
    '''
    def learning(self, X, t, learning_rate=0.01, iters=1000, batch_size=100):
        loss_history = []  # 记录每次迭代的损失值
        acc_history = [] # 记录一次 epoch 的精度值
        grads = {}  # 记录参数 W 和 B 的梯度值
        train_size = X.shape[0]
        iter_per_epoch = max(train_size / batch_size, 1) # 计算需要多少轮 epoch

        for i in range(iters):
            # 每次只随机看 batch_size 个数据
            batch_mask = np.random.choice(train_size, batch_size)
            X_batch = X[batch_mask]
            t_batch = t[batch_mask]

            # 梯度下降算法
            grads = self.gradient_descent(X_batch, t_batch)
            for key in ('W1', 'B1', 'W2', 'B2'):
                self.params[key] -= learning_rate * grads[key]

            # 每经过一轮 epoch，就代表所有数据都被“查看过了”，计算识别精度
            if i % iter_per_epoch == 0:
                train_acc = self.accuracy(X, t)
                acc_history.append(train_acc)
                print(f"epoch {int(i / iter_per_epoch)} (iter {i}): train accuracy = {train_acc}")

            # 计算损失值
            loss = self.loss(X_batch, t_batch)
            # print(f"iter {i}: loss = {loss}")
            loss_history.append(loss)

        return acc_history, loss_history

    # 神经网络参数的精度评价
    def accuracy(self, X, t):
        y = self.predict(X)
        '''
        axis=0：返回每一列最大值的索引；axis=1：返回每一行最大值的索引
        axis=None：降为一维数组后，返回最大值的索引
        '''
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(X.shape[0])
        return accuracy

# ============================= 其它函数 other ===================================

# 绘制折线图
def draw_scatter(x, y, title):
    plt.xlabel("X-axis", size=15)
    plt.ylabel("Y-axis", size=15)
    plt.title(title, size=20)
    plt.plot(x, y, linestyle="-")

# ============================= 主程序 main ===================================

if __name__ == '__main__':
    # 获取训练集和测试集（一维化、归一化、采用one-hot标签）
    # x_train: (60000,784), t_train: (60000,10)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)
    learning_rate = 0.1  # 学习率
    iters = 10000  # 迭代次数
    batch_size = 100 # 每个epoch查看多少个数据

    loss_history = []
    acc_history = []
    params = {}

    # 初始化神经网络的参数，输入层 784 个神经元，隐藏层 50 个神经元，输出层 10 个神经元
    network = TwoLayerNet(input_size = x_train.shape[1],
                          hidden_size = 50,
                          output_size = t_train.shape[1])

    # 2 层神经网络的学习
    acc_history, loss_history = network.learning(X = x_train,
                                       t = t_train,
                                       learning_rate = learning_rate,
                                       iters = iters,
                                       batch_size = batch_size)

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size / batch_size, 1)
    x = list(range(int(iters / iter_per_epoch) + 1))
    draw_scatter(x, acc_history, "Training Accuracy")
    plt.show()

    x = list(range(iters))
    draw_scatter(x, loss_history, "Loss History")
    plt.show()
