import numpy as np

# sigmoid 函数
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# softmax 函数
def softmax_function(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 改良的 softmax 函数（防止在指数运算时发生溢出）
def softmax_function_trick(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# ReLU 函数
def relu_function(x):
    return np.maximum(0, x)

# 线性激活函数（恒等函数）
def linear_activation_function(x):
    return x

# 初始化配置各个神经元的参数
def init_network():
    network = {} # 字典类型

    # 隐藏层第 1 层（layer 1）：一共 4 个神经元
    network['W1'] = np.array([[0.1, 0.2, 0.3, 0.4],
                              [0.5, 0.6, 0.7, 0.8]])
    network['B1'] = np.array([[0.1, 0.2, 0.3, 0.4]])

    # 隐藏层第 2 层（layer 2）：一共 5 个神经元
    network['W2'] = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                              [0.6, 0.7, 0.8, 0.9, 1.0],
                              [0.1, 0.2, 0.3, 0.4, 0.5],
                              [0.6, 0.7, 0.8, 0.9, 1.0]])
    network['B2'] = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

    # 隐藏层第 3 层（layer 3）：一共 3 个神经元
    network['W3'] = np.array([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.6],
                              [0.7, 0.8, 0.9],
                              [0.6, 0.7, 0.8],
                              [0.1, 0.2, 0.3]])
    network['B3'] = np.array([[0.1, 0.2, 0.3]])

    # 隐藏层第 4 层（layer 4）：一共 1 个神经元
    network['W4'] = np.array([[0.1],
                              [0.2],
                              [0.3]])
    network['B4'] = np.array([[0.1]])

    return network

# 神经元的内部实现：输入A，权重W，偏置B，激活函数g()，输出A_out
def dense(A, W, B, g):
	Z = np.matmul(A, W) + B # 这里是矩阵乘法，而非点乘
	A_out = g(Z)
	return A_out

# 神经网络的搭建
def predict(network, X):
    W1, W2, W3, W4 = network['W1'], network['W2'], network['W3'], network['W4']
    B1, B2, B3, B4 = network['B1'], network['B2'], network['B3'], network['B4']

    A1 = dense(X, W1, B1, sigmoid_function) # layer 1
    A2 = dense(A1, W2, B2, sigmoid_function) # layer 2
    A3 = dense(A2, W3, B3, sigmoid_function) # layer 3
    A4 = dense(A3, W4, B4, linear_activation_function) # layer 4

    return A4

# 从这里开始执行
if __name__ == '__main__':
    network = init_network() # 配置神经网络的参数
    X = np.array([[1.0, 0.5]]) # 输入层（layer 0）
    Y = predict(network, X) # 输出层（layer 4）
    print(Y)
