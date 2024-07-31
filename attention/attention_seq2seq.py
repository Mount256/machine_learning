# coding: utf-8
import sys
import os
sys.path.append('..')
from dataset import sequence
from trainer import Trainer
from trainer import eval_seq2seq
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

'''
Sigmoid 函数
其导数为 y' = y * (1 - y)
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ============================= 基本层的实现 =============================
# 以后实现的层都有 params（eg.权重和偏置）和 grads 实例变量，由于可能有多个参数，所以均使用列表存储
# 所有的层都有 forward() 和 backward() 方法

class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx

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
        out = W[idx] # 根据 idx 提取对应的权重行
        return out # 返回提取后的权重值

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0 # 初始化为零矩阵

        np.add.at(dW, self.idx, dout)

'''
LSTM 层：Long Short-term Memory（长短期记忆）的缩写，意思是长时间维持短期记忆
【正向传播】如下：
- 输入端：c_prev(N, H), h_prev(N, H), x_t(N, D)
- 权重：Wx(D, 4H), Wh(H, 4H), b(N, 4H)
- 仿射变换：
  - f = sigmoid(x_t Wx[:, H] + h_prev Wh[:, H] + b), f(N, H)
  - g = tanh(x_t Wx[:, H:2*H] + h_prev Wh[:, H:2*H] + b), g(N, H)
  - i = sigmoid(x_t Wx[:, 2*H:3*H] + h_prev Wh[:, 2*H:3*H] + b), i(N, H)
  - o = sigmoid(x_t Wx[:, 3*H:4*H] + h_prev Wh[:, 3*H:4*H] + b), o(N, H)
- 输出端：
  - c_next = f * c_prev + g * i (* 为哈达玛积), c_next(N, H)
  - h_next = o * tanh(c_next) (* 为哈达玛积), h_next(N, H)
【反向传播】直接见如下代码（以及《深度学习进阶：自然语言处理》P242图）
'''
class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        # 4 个仿射变换整合为一个式子进行，A(N, 4H)
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        # slice
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)
        dh_s = o * dh_next
        dc_s = dc_next + dh_s * (1 - tanh_c_next**2)

        dc_prev = f * dc_s
        df = c_prev * dc_s
        dg = i * dc_s
        di = g * dc_s
        do = tanh_c_next * dh_next

        df *= f * (1 - f)
        dg *= (1 - g**2)
        di *= i * (1 - i)
        do *= o * (1 - o)

        # np.hstack() 在水平方向上将矩阵拼接起来
        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)
        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev, dc_prev

'''
Weight Sum 层

参数说明：
- hs：单词向量（编码器输出得来）
- a：各个单词重要度的权重（概率分布）

正向传播如下图所示：
                                  hs(N,T,H)
                                  |
                                  |
                                  V
a(N,T) --Repeat--> ar(N,T,H) --Product--> t(N,T,H) --Sum--> c(N,H)

反向传播如下图所示：
                                  dhs(N,T,H)
                                  ^
                                  | (dt * ar)
                                  |
da(N,T) <--Sum-- dar(N,T,H) <--Product-- dt(N,T,H) <--Repeat-- dc(N,H)
                              (dt * hs)
'''
class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape

        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dhs = dt * ar
        dar = dt * hs
        da = np.sum(dar, axis=2)

        return dhs, da

'''
Attention Weight 层

参数说明：
- hs：单词向量（编码器输出得来）
- h：解码器的 LSTM 层的隐藏状态向量

正向传播如下图所示：
                                  hs(N,T,H)
                                  |
                                  |
                                  V
h(N,T) --Repeat--> hr(N,T,H) --Product--> t(N,T,H) --Sum--> s(N,T) --Softmax--> a(N,T)

反向传播如下图所示：
                                  dhs(N,T,H)
                                  ^
                                  | (dt * hr)
                                  |
dh(N,T) <--Sum-- dhr(N,T,H) <--Product-- dt(N,T,H) <--Repeat-- ds(N,T) <--Softmax-- da(N,T)
                              (dt * hs)
'''
class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        t = hs * hr
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s)

        self.cache = (hs, hr)
        return a

    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, axis=1)

        return dhs, dh

'''
Attention 层的结构：
                hs                hs
                |                 |
                |                 |
                V                 V
h --> Attention Weight --> a --> Weight Sum --> c
'''
class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out

    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh

# ============================= Time 层的实现 =============================
# 以后实现的层都有 params（eg.权重和偏置）和 grads 实例变量，由于可能有多个参数，所以均使用列表存储
# 所有的层都有 forward() 和 backward() 方法

'''
Time Embedding 层：由多个 Embedding 层组成，每个层之间不用连接，单独处理即可
- 输入端：x_s(N, T)，其中每个 Embedding 层输入为 x_i(N, )
- 参数：每个 Embedding 层的权重大小为 W(V, D)
- 输出端：将 x_s 作为行索引提取 W，形成 out(N, T, D)
  - 其中：每个 Embedding 层将 x_i(N, ) 作为行索引提取 W，输出为 out_i(N, D)
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
Time LSTM 层：由多个 LSTM 层组成
正向传播过程如下：  
-----------------------------------------------------------------------------------------------------------------
x_s(N,T,D)=                    x_0(N,D)              x_1(N,D)              x_2(N,D)  x_T(N,D)  
                               |                       |                     |         |
                               |                       |                     |         |
                               V                       V                     V         V
上一个TimeLSTM层 --c,h(N,H)--> LSTM_0 --c,h_0(N,H)--> LSTM_1 --c,h_1(N,H)--> LSTM_2 ... LSTM_T --c,h_T(N,H)--> 下一个TimeLSTM层
                                  |                  |                     |           |
                                  |                  |                     |           |
                                  V                  V                     V           V
h_s(N,T,H)=                      h_0(N,H)           h_1(N,H)             h_2(N,H)    h_T(N,H)   
-----------------------------------------------------------------------------------------------------------------
反向传播过程如下：
-----------------------------------------------------------------------------------------------------------------
dx_s(N,T,D)=                   dx_0(N,D)                dx_1(N,D)                  dx_2(N,D) dx_T(N,D) 
                                 ^                           ^                          ^         ^ 
                                 |                           |                          |         |
                                 |                           |                          |         |
上一个TimeLSTM层 <--dc,dh(N,H)-- LSTM_0 <--dc_0,dh_0(N,H)-- LSTM_1 <--dc_1,dh_1(N,H)-- LSTM_2 ... LSTM_T
                                ^                             ^                         ^          ^
                                |                             |                         |          |
                                |                             |                         |          |
dh_s(N,T,H)=                   dhs_0(N,H)                 dhs_1(N,H)               dhs_2(N,H) dhs_T(N,H)   
-----------------------------------------------------------------------------------------------------------------     
RNN 层各权重矩阵的尺寸：Wx(D, 4H), Wh(H, 4H), b(N, 4H)  
'''
class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    # 设定隐藏状态
    def set_state(self, h, c=None):
        self.h, self.c = h, c

    # 重设隐藏状态
    def reset_state(self):
        self.h, self.c = None, None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')  # 创建一个空矩阵（可理解为预先分配指定大小的内存）

        # 若隐藏状态为 False，则第一个 LSTM 层的隐藏状态被初始化为零矩阵
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        # 连接每个 LSTM 层（一共 T 层）并正向传播
        for t in range(T):
            layer = LSTM(*self.params)  # *号表示列表解构：将列表中的元素分配给几个变量
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)  # 此 LSTM 层输出的 h 值作为后一个 LSTM 层的输入
            hs[:, t, :] = self.h  # 最后一个 LSTM 层的输出作为整个 TimeLSTM 层的输出
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape  # N：批大小，D：输入向量的维数，T：时序数据的个数
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')  # 创建一个空矩阵（可理解为预先分配指定大小的内存）
        dh, dc = 0, 0
        grads = [0, 0, 0]

        # 连接每个 LSTM 层（一共 T 层）并反向传播
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh  # 隐藏状态的梯度值存放在成员变量 dh 中

        return dxs

'''
Time Attention 层：由多个 Attention 层组成
正向传播过程如下（Attention简写为A），这里仅展示了一个 Attention 层的情况，实际上是由多个这样的 Attention 层组成。

out[:,0,:]
^
|
|
A0
^
|
|
hs_enc,
hs_dec[:,0,:]
'''
class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec


'''
Time Affine 层：由多个 Affine 层组成，每个层之间不用连接，单独处理即可
- 输入端：x_s(N, T, H)，其中每个 Affine 层输入为 x_i(N, H)
- 参数：每个 Affine 层均为 W(H, V)，b(V, )
- 输出端：大小为 (N, T, V)，其中每个 Affine 层输出大小为 (N, V)
'''
class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1) # 输入端展开为二维矩阵
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1) # 恢复原形状

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1) # 输入端展开为二维矩阵
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

# ============================= 神经网络的定义 =============================
'''
Encoder 的构成如下：
字符向量 xs --> Time Embedding --> Time LSTM --> （丢弃）
                                   |
                                   ----> h （所有 LSTM 层均输出隐藏状态）
'''
class AttentionEncoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        # 小批量样本个数，向量维数，隐藏层维数
        V, D, H = vocab_size, wordvec_size, hidden_size

        # 初始化权重
        embed_W = (np.random.randn(V, D) / 100).astype('f')  # 正态分布初始值
        lstm_Wx = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype('f')  # Xaiver 初始值
        lstm_Wh = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype('f')  # Xaiver 初始值
        lstm_b = np.zeros(4 * H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs

    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout

'''
Decoder 的构成如下：
                                                      (dec_hs)
                                             ---------------------------
                                             |                          |
                                             |                          V
字符向量 xs --> Time Embedding --> Time LSTM --> Time Attention --> Time Affine --> (Time softmax with Loss --> 损失)
                                    ^                ^          (c)            (score)
                                    |                |
                              enc_hs 的最后一行     enc_hs
                              （Encoder 输出的隐藏状态）
'''
class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        # 小批量样本个数，向量维数，隐藏层维数
        V, D, H = vocab_size, wordvec_size, hidden_size

        # 初始化权重
        embed_W = (np.random.randn(V, D) / 100).astype('f')  # 正态分布初始值
        lstm_Wx = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype('f')  # Xaiver 初始值
        lstm_Wh = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype('f')  # Xaiver 初始值
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (np.random.randn(H + H, V) / np.sqrt(H + H)).astype('f')  # Xaiver 初始值
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.attention, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, enc_hs):
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        dec_hs = self.lstm.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H = dout.shape
        H = H // 2

        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:] # 注意顺序！
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)

        return denc_hs

    # 生成字符
    # h：编码器输出的隐藏状态，start_id：第一个字符 ID，sample_size：生成字符的数量
    def generate(self, enc_hs, start_id, sample_size):
        sampled = []  # 存放已被采样的字符 ID
        sample_id = start_id  # 存放正在被采样的字符 ID
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape(1, 1)  # 由于网络的输入必须为矩阵，所以要预先将 sample_id 转化为矩阵形式
            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())  # 选取得分最高的字符 ID
            sampled.append(int(sample_id))

        return sampled

'''
seq2seq 的构成如下：
                            xs          ts（监督数据）
                            |            |
                            |            |
                            V            V
xs --> Encoder --> h --> Decoder --> Time softmax with Loss --> 损失
'''
class AttentionSeq2seq:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        # 小批量样本个数，向量维数，隐藏层维数
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(V, D, H)
        self.decoder = AttentionDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:] # ???

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    # 生成字符
    # h：编码器输出的隐藏状态，start_id：第一个字符 ID，sample_size：生成字符的数量
    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled

    # 加载训练好的参数
    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        if '/' in file_name:
            file_name = file_name.replace('/', os.sep)

        if not os.path.exists(file_name):
            raise IOError('No file: ' + file_name)

        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]
        for i, param in enumerate(self.params):
            param[...] = params[i]

    # 保存训练好的参数
    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        params = [p.astype(np.float16) for p in self.params]
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

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

'''
Adam (http://arxiv.org/abs/1412.6980v8)
'''
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

# ============================= 主程序 main ===================================
if __name__ == '__main__':
    # 读入数据集
    (x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
    char_to_id, id_to_char = sequence.get_vocab()
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1] # 反转输入数据后的效果更佳！！！

    # 设定超参数
    batch_size = 128  # (N=128)
    vocab_size = len(char_to_id)  # (V=59)
    wordvec_size = 16  # (D=16)
    hidden_size = 256  # 隐藏状态向量的元素个数 (H=256)
    max_epoch = 10
    max_grad = 5.0  # 用于梯度裁剪

    # 生成模型
    model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    acc_list = []
    for epoch in range(max_epoch):
        trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

        correct_num = 0
        for i in range(len(x_test)):
            question, correct = x_test[[i]], t_test[[i]]
            verbose = i < 10
            correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose)
        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)
        print("val acc %.3f%%" % (acc * 100))

    model.save_params()

    # 绘制图形
    x = np.arange(len(acc_list))
    plt.plot(x, acc_list, marker='o')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(-0.05, 1.05)
    plt.show()
