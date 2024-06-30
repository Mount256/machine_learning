# coding: utf-8
import sys
sys.path.append('..')
import pickle
from dataset import ptb
from rnnlm_trainer import RnnlmTrainer
from rnnlm_trainer import eval_perplexity

GPU = False # 是否使用 GPU 的标志

if GPU:
    import cupy as np
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')
else:
    import numpy as np

# ============================= 公用函数 =============================
def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)

def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)

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

        if GPU:
            import cupyx
            cupyx.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None

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
    def set_state(self, h, c):
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
        if GPU:
            mask = to_gpu(mask)

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
                                                          t_s(N,T)（正确解标签）
                                                            |
                                                            |
                                                            V
w_s --> Time Embedding --> Time LSTM --> Time Affine --> Time Softmax with Loss --> loss
(N,T)                 (N,T,D)       (N,T,H)         (N,T,V)                    

Xaiver 初始值：在上一层有 n 个节点的情况下，使用标准差为 1/sqrt(n) 的分布作为初始值
'''
class RNNLM:
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        # 小批量样本个数，向量维数，隐藏层维数
        V, D, H = vocab_size, wordvec_size, hidden_size

        # 初始化权重
        embed_W = (np.random.randn(V, D) / 100).astype('f')  # 正态分布初始值
        lstm_Wx = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype('f')  # Xaiver 初始值
        lstm_Wh = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype('f')  # Xaiver 初始值
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype('f')  # Xaiver 初始值
        affine_b = np.zeros(V).astype('f')

        self.layers = [TimeEmbedding(embed_W),
                       TimeLSTM(lstm_Wx, lstm_Wh, lstm_b),
                       TimeAffine(affine_W, affine_b)]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]  # 专门设置一个 RNN 成员变量保存 TimeRNN 层

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()

    def save_params(self, file_name='RNNLM.pkl'):
        params = [p.astype(np.float16) for p in self.params]
        if GPU:
            params = [to_cpu(p) for p in params]
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name='RNNLM.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        params = [p.astype('f') for p in params]
        if GPU:
            params = [to_gpu(p) for p in params]
        for i, param in enumerate(self.params):
            param[...] = params[i]

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
    # 设定超参数
    batch_size = 20  # (N=20)
    wordvec_size = 100  # (D=100)
    hidden_size = 100  # LSTM 层的隐藏状态向量的元素个数 (H=100)
    time_size = 35  # Truncated BPTT 的时间跨度大小 (T=35)
    lr = 20.0  # 学习率
    max_epoch = 4
    max_grad = 0.25

    # 读入训练集数据
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_t, _, _ = ptb.load_data('test')
    xs = corpus[:-1]  # 输入
    ts = corpus[1:]  # 输出（监督标签）
    vocab_size = len(word_to_id)
    if GPU:
        xt, ts = to_gpu(xs), to_gpu(ts)

    # 生成模型
    model = RNNLM(vocab_size, wordvec_size, hidden_size)  # V=vocab_size, D=100, H=100
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    # 模型训练
    trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
    trainer.plot(ylim=(0, 500))

    # 基于测试数据进行困惑度评价
    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_t)
    print('test perplexity: ', ppl_test)

    # 保存参数
    model.save_params()
