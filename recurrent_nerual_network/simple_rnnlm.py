# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
# from rnnlm_trainer import RnnlmTrainer

# ============================= 公用函数 =============================
'''
Softmax 函数
'''
def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

# ============================= 基本层的实现 =============================
# 以后实现的层都有 params（eg.权重和偏置）和 grads 实例变量，由于可能有多个参数，所以均使用列表存储
# 所有的层都有 forward() 和 backward() 方法

'''
Embedding 层
'''
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None # 保存需要提取的行的索引（单词ID）

    def forward(self, idx):
        W, = self.params # 这里的逗号为了取出权重 W 而不是 [W]
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0 # 初始化为零矩阵

        if GPU:
            import cupyx
            cupyx.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None

'''
RNN 层正向传播：h_t = \tanh(h_{t-1}W_h + x_tW_x + b)
'''
class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dtanh = dh_next * (1 - h_next ** 2) # 请查阅 tanh 函数的导数公式
        db = np.sum(dtanh, axis=0)
        dWh = np.dot(h_prev.T, dtanh)
        dh_prev = np.dot(dtanh, Wh.T)
        dWx = np.dot(x.T, dtanh)
        dx = np.dot(dtanh, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dh_prev, dx

# ============================= Time 层的实现 =============================
# 以后实现的层都有 params（eg.权重和偏置）和 grads 实例变量，由于可能有多个参数，所以均使用列表存储
# 所有的层都有 forward() 和 backward() 方法

'''
TimeRNN 层：由多个 RNN 层组成
正向传播过程如下：  
-----------------------------------------------------------------------------------------------------------------
x_s(N,T,D)=                    x_0(N,D)          x_1(N,D)              x_2(N,D)  x_T(N,D)  
                               |                 |                     |         |
                               |                 |                     |         |
                               V                 V                     V         V
上一个TimeRNN层 --h(N,H)--> RNN_0 --h_0(N,H)--> RNN_1 --h_1(N,H)--> RNN_2 ... RNN_T --h_T(N,H)--> 下一个TimeRNN层
                                  |                  |                     |         |
                                  |                  |                     |         |
                                  V                  V                     V         V
h_s(N,T,H)=                      h_0(N,H)           h_1(N,H)             h_2(N,H)  h_T(N,H)   
-----------------------------------------------------------------------------------------------------------------
反向传播过程如下：
-----------------------------------------------------------------------------------------------------------------
dx_s(N,T,D)=                 dx_0(N,D)         dx_1(N,D)             dx_2(N,D) dx_T(N,D) 
                               ^                 ^                     ^         ^ 
                               |                 |                     |         |
                               |                 |                     |         |
上一个TimeRNN层 <--dh(N,H)-- RNN_0 <--dh_0(N,H)-- RNN_1 <--dh_1(N,H)-- RNN_2 ... RNN_T
                               ^                  ^                     ^          ^
                               |                  |                     |          |
                               |                  |                     |          |
dh_s(N,T,H)=                  dhs_0(N,H)         dhs_1(N,H)            dhs_2(N,H) dhs_T(N,H)   
-----------------------------------------------------------------------------------------------------------------       
RNN 层各矩阵的尺寸如下：
- 权重值：
  - Wx：（D, H）
  - Wh：（H, H）
  - b：（N, H）     
- 输入端：
  - h_{t-1}：（N, H）
  - x_t：（N, D）
- 输出端：
  - h_t：（N, H）
'''
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b] # 这些均为 RNN 层要设置的参数
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h = None # 保存调用 forward() 方法时最后一个 RNN 层的隐藏状态
        self.dh = None # 保存调用 backward() 方法时前一个块的隐藏状态的梯度
        self.stateful = stateful # 记录 TimeRNN 层的隐藏状态，True 表示保存上一时刻的隐藏状态
        '''
        若调用 forward() 方法，则成员变量 h 存放最后一个 RNN 层的隐藏状态。
        若 stateful = True，则下一次调用 forward() 方法时，刚才的成员变量 h 的值将被继续使用；
        若 stateful = False，则成员变量 h 的值将被重置为零向量。
        '''

    # 设定隐藏状态
    def set_state(self, h):
        self.h = h

    # 重设隐藏状态
    def reset_state(self):
        self.h = None

    # 正向传播的示意图见上方
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape # N：批大小，D：输入向量的维数，T：时序数据的个数
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f') # 创建一个空矩阵（可理解为预先分配指定大小的内存）

        # 若隐藏状态为 False，则第一个 RNN 层的隐藏状态被初始化为零矩阵
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        # 连接每个 RNN 层（一共 T 层）并正向传播
        for t in range(T):
            layer = RNN(*self.params) # *号表示列表解构：将列表中的元素分配给几个变量
            self.h = layer.forward(xs[:, t, :], self.h) # 此 RNN 层输出的 h 值作为后一个 RNN 层的输入
            hs[:, t, :] = self.h # 最后一个 RNN 层的输出作为整个 TimeRNN 层的输出
            self.layers.append(layer)

        return hs

    # 反向传播的示意图见上方
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape  # N：批大小，D：输入向量的维数，T：时序数据的个数
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f') # 创建一个空矩阵（可理解为预先分配指定大小的内存）
        dh = 0
        grads = [0, 0, 0]

        # 连接每个 RNN 层（一共 T 层）并反向传播
        for t in reversed(range(T)):
            layer = self.layer[t]
            # 每个 RNN 层正向传播的输出由两个分叉，则反向传播时流向 RNN 层的是求和后的梯度
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh # 隐藏状态的梯度值存放在成员变量 dh 中

        return dxs

'''
Time Embedding 层：由多个 Embedding 层组成
'''
class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None

'''
Time Affine 层：由多个 Affine 层组成
'''
class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

'''
Time SoftmaxWithLoss 层：由多个 SoftmaxWithLoss 层组成
'''
class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 在监督标签为one-hot向量的情况下
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # 按批次大小和时序大小进行整理（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # 与ignore_label相应的数据将损失设为0
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # 与ignore_label相应的数据将梯度设为0

        dx = dx.reshape((N, T, V))

        return dx

# ============================= RNNLM 的类定义 =============================
'''
RNNLM 层的构成如下：
                                                          t_s（正确解标签）
                                                            |
                                                            |
                                                            V
w_s --> Time Embedding --> Time RNN --> Time Affine --> Time Softmax with Loss --> loss

Xaiver 初始值：在上一层有 n 个节点的情况下，使用标准差为 1/sqrt(n) 的分布作为初始值
'''
class SimpleRNNLM:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        # 小批量样本个数，向量维数，隐藏层维数
        V, D, H = vocab_size, wordvec_size, hidden_size

        # 初始化权重
        embed_W = (np.random.randn(V, D) / 100).astype('f') # 正态分布初始值
        rnn_Wx = (np.random.randn(D, H) / np.sqrt(D)).astype('f') # Xaiver 初始值
        rnn_Wh = (np.random.randn(H, H) / np.sqrt(H)).astype('f') # Xaiver 初始值
        rnn_b = np.zeros(H).astype('f')
        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype('f') # Xaiver 初始值
        affine_b = np.zeros(V).astype('f')

        self.layers = [TimeEmbedding(embed_W),
                       TimeRNN(rnn_Wx, rnn_Wh, rnn_b),
                       TimeAffine(affine_W, affine_b)]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1] # 专门设置一个 RNN 成员变量保存 TimeRNN 层

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.forward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()

# ============================= 优化器的定义 =============================
'''
随机梯度下降法（Stochastic Gradient Descent）
'''
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

# ============================= 主程序 main ===================================
if __name__ == '__main__':
    pass
