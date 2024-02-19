# coding: utf-8

import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist

# ============================= activation functions ===================================

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

# ============================= neural network ===================================

# 神经元的内部实现：输入A，权重W，偏置B，激活函数g()，输出A_out
def dense(A, W, B, g):
	Z = np.matmul(A, W) + B # 这里是矩阵乘法，而非点乘
	A_out = g(Z)
	return A_out

# 初始化配置各个神经元的参数，可直接导入记录了神经网络参数的pickle文件
def init_network(filename):
    with open(filename, 'rb') as f:
        network = pickle.load(f)
    return network

# 神经网络的搭建
def predict(network, X):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['b1'], network['b2'], network['b3']
    A1 = dense(X, W1, B1, sigmoid_function) # layer 1
    A2 = dense(A1, W2, B2, sigmoid_function) # layer 2
    A3 = dense(A2, W3, B3, softmax_function_trick) # layer 3
    return A3

# 获取训练集和测试集
def get_data():
    # 训练集数据和结果，测试集数据和结果
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_train, t_train, x_test, t_test

# 模型评估
def assessment():
    pass

# ============================= main ===================================

if __name__ == '__main__':
    network = init_network('sample_weight.pkl') # 配置神经网络的参数
    _, _, X, T = get_data() # X：测试集数据，T：测试集正确结果
    accuracy_cnt = 0 # 记录推理正确的个数

    for i in range(X.shape[0]): # X.shape[0] 即为测试集数据个数
        Y = predict(network, X[i])  # 对测试集每个数据进行推理，得到 10 个概率数值的一维数组
        # print(Y)
        # axis=0：返回每一列最大值的索引；axis=1：返回每一行最大值的索引
        # axis=None：降为一维数组后，返回最大值的索引
        p = np.argmax(Y, axis=None) # 返回概率最大的索引
        if p == T[i]: # 如果推理结果与测试集结果相同，说明推理正确
            accuracy_cnt += 1

    print(f"accuracy: {float(accuracy_cnt) / X.shape[0]}") # 精度结果：93.52%
