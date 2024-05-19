# coding: utf-8

import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from collections import OrderedDict

# ============================= 图像与矩阵的相互转换 ===================================
'''
将图像转化为矩阵（im2col = image to column）
    input_data: 4维矩阵，（数据量，通道，高，长）
    filter_h: 滤波器的高
    filter_w: 滤波器的宽
    stride: 步幅
    pad: 填充
    返回 col : 2维数组
'''
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

'''
将矩阵转化为图像（col2im = column to image）
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h: 滤波器的高
    filter_w: 滤波器的宽
    stride: 步幅
    pad: 填充
    返回 img : 2维图像
'''
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

# ============================= 损失函数 Loss functions ===================================

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

# ============================= 激活函数 Activation functions ===================================
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

# =============== Sigmoid 函数 ===============
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

        self.X = None
        self.original_X_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.dB = None

    # 前向传播
    def forward(self, X):
        # 对应张量
        self.original_X_shape = X.shape
        X = X.reshape(X.shape[0], -1)
        self.X = X

        out = np.dot(self.X, self.W) + self.B
        return out

    # 反向传播
    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.dB = np.sum(dout, axis=0)

        dX = dX.reshape(*self.original_X_shape)  # 还原输入数据的形状（对应张量）
        return dX

# =============== Softmax with loss ===============
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

        if self.t.size == self.y.size:  # 监督数据是 one-hot-vector 的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

# =============== 卷积层 Convolution ===============
class Convolution:
    def __init__(self, W, B, stride=1, pad=0):
        self.W = W
        self.B = B
        self.stride = stride # 应用滤波器的间隔
        self.pad = pad # 幅度为 pad 的填充

        self.dW = None
        self.dB = None

        self.X = None
        self.col = None
        self.col_W = None

    # 前向传播
    def forward(self, X):
        # FN 个滤波器，每个滤波器有 C 个通道，每个通道高度为 FH，宽度为 FW
        FN, C, FH, FW = self.W.shape
        N, C, H, W = X.shape
        OH = 1 + int((H + 2 * self.pad - FH) / self.stride)
        OW = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(X, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T # 滤波器的展开，若4维矩阵共750个元素，reshape(10, -1) 转换为 (10, 75) 的数组
        out = np.dot(col, col_W) + self.B
        out = out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)

        self.X = X
        self.col = col
        self.col_W = col_W

        return out

    # 反向传播
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.dB = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dX = col2im(dcol, self.X.shape, FH, FW, self.stride, self.pad)

        return dX

# =============== 池化层 Pooling ===============
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride # 应用滤波器的间隔
        self.pad = pad # 幅度为 pad 的填充

        self.X = None
        self.arg_max = None

    # 前向传播
    def forward(self, X):
        N, C, H, W = X.shape
        OH = 1 + int((H - self.pool_h) / self.stride)
        OW = 1 + int((W - self.pool_w) / self.stride)

        # 展开输入数据
        col = im2col(X, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 求各行的最大值
        self.arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        # 转换为合适的输出大小
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

        self.X = X
        return out

    # 反向传播
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dX = col2im(dcol, self.X.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dX

# ============================= 卷积神经网络 Convolutional Neural Network ===================================
class ConvNet: # CNN 构成：Conv - ReLU1 - Pooling - Affine1 - ReLU2 - Affine2 - Softmax
    '''
    初始化配置各个神经元的参数：
    input_dim：输入数据的维度（通道[0]，高[1]，长[2]）
    conv_param：卷积层的超参数
        filter_num: 滤波器的数量
        filter_size: 滤波器的大小
        stride: 步幅
        pad: 填充
    hidden_size：隐藏层（Affine）的神经元数量
    output_size：输出层（Affine）的神经元数量
    weight_init_std：初值权重值的标准差
    loss_f：损失函数
    '''
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01, loss_f=cross_entropy_error):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        self.params = {}  # 初始化权重，字典类型
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['B1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['B2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['B3'] = np.zeros(output_size)

        self.layers = OrderedDict()  # 生成各种层，有序字典类型
        self.layers['Conv'] = Convolution(W=self.params['W1'], B=self.params['B1'], stride=filter_stride, pad=filter_pad)
        self.layers['ReLU1'] = ReLU()
        self.layers['Pooling'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['B2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['B3'])
        self.lastLayer = SoftmaxWithLoss(loss_f)

    # 卷积神经网络的前向传播（不包含 softmax 分类）
    def predict(self, X):
        for layer in self.layers.values():  # values()：获取字典所有的值，返回一个列表
            X = layer.forward(X)
        return X

    # 计算损失/误差值
    def loss(self, X, t):
        y = self.predict(X)  # 先进行推理（前向传播）
        return self.lastLayer.forward(y, t)  # 后用推理结果与正确解标签计算损失值

    # 误差反向传播法求梯度
    def gradient_descent(self, X, t):
        # step1.前向传播
        self.loss(X, t)

        # step2.误差反向传播
        # 从最后一层开始反向传播
        dout = 1
        dout = self.lastLayer.backward(dout)
        # 从后往前依次反向传播
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv'].dW
        grads['B1'] = self.layers['Conv'].dB
        grads['W2'] = self.layers['Affine1'].dW
        grads['B2'] = self.layers['Affine1'].dB
        grads['W3'] = self.layers['Affine2'].dW
        grads['B3'] = self.layers['Affine2'].dB

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
            for key in ('W1', 'B1', 'W2', 'B2', 'W3', 'B3'):
                self.params[key] -= learning_rate * grads[key]

            # 每经过一轮 epoch，就代表所有数据都被“查看过了”，计算识别精度
            if i % iter_per_epoch == 0:
                train_acc = self.accuracy(X, t, batch_size)
                acc_history.append(train_acc)
                print(f"epoch {int(i / iter_per_epoch)} (iter {i}): train accuracy = {train_acc}")

            # 计算损失值
            loss = self.loss(X_batch, t_batch)
            # print(f"iter {i}: loss = {loss}")
            loss_history.append(loss)

        return acc_history, loss_history

    # 卷积神经网络参数的精度评价
    def accuracy(self, X, t, batch_size):
        if t.ndim != 1: t = np.argmax(t, axis=1)
        acc = 0.0

        for i in range(int(X.shape[0] / batch_size)):
            tx = X[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / X.shape[0]

# ============================= 其它函数 Other ===================================

# 绘制折线图
def draw_scatter(x, y, title):
    plt.xlabel("X-axis", size=15)
    plt.ylabel("Y-axis", size=15)
    plt.title(title, size=20)
    plt.plot(x, y, linestyle="-")

# ============================= 主程序 Main ===================================

if __name__ == '__main__':
    # 获取训练集和测试集（一维化、归一化、采用one-hot标签）
    # x_train: (60000,784), t_train: (60000,10)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True, one_hot_label=False)
    # 为加快运行速度而减少样本数
    x_train, t_train = x_train[:5000], t_train[:5000]
    x_test, t_test = x_test[:1000], t_test[:1000]

    learning_rate = 0.1  # 学习率
    iters = 1000  # 迭代次数
    batch_size = 100 # 每个epoch查看多少个数据

    # 初始化卷积神经网络的参数
    network = ConvNet(input_dim=(1, 28, 28),
                      conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                      hidden_size=100,
                      output_size=10,
                      weight_init_std=0.01)

    # 卷积神经网络的学习
    acc_history, loss_history = network.learning(X=x_train,
                                                 t=t_train,
                                                 learning_rate=learning_rate,
                                                 iters=iters,
                                                 batch_size=batch_size)

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size / batch_size, 1)
    x = list(range(int(iters / iter_per_epoch)))
    draw_scatter(x, acc_history, "Training Accuracy")
    plt.show()

    x = list(range(iters))
    draw_scatter(x, loss_history, "Loss History")
    plt.show()
