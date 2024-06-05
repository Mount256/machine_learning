# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
import collections
import os
import pickle
from dataset import ptb
from trainer import Trainer

'''
CBOW 模型使用的神经网络的输入是上下文，要预测（输出）的是被上下文包围的单词，
这跟英语的【完形填空】是一样的
'''

GPU = False # 是否使用 GPU 的标志

# ============================= 公用函数 =============================
'''
生成上下文和目标词
    :param corpus: 语料库（单词ID列表）
    :param window_size: 窗口大小（当窗口大小为1时，左右各1个单词为上下文）
    :return: 
      - contexts 大小为 (len(corpus)-window_size*2, window_size*2)
      - target 大小为 (len(corpus)-window_size*2, window_size)
'''
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)

'''
Sigmoid 函数
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

'''
交叉熵误差函数
'''
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 在监督标签为one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# ============================= 负例采样器的定义 =============================
class UnigramSampler:
    '''
    __init__：计算每个单词的概率分布
    参数说明：
      - corpus: 语料库转化为以单词ID表示的列表（一维或二维的NumPy数组）
      - power：对概率分布取幂次方，是为了防止低频单词被忽略
      - sample_size：一个负例样本里有多少个样本
    举例：corpus = [1, 2, 0, 3, 4, 4, 2]
    单词 ID 小批量样本 target = [1, 2, 4, 0], sample_size = 2
    则 batch_size = 4, negative_sample 大小为 (4, 2)
    现负例随机采样得到：
    [[1, 0],
    [2, 4],
    [0, 2],
    [2, 1]]
    '''
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None # （不重复的）单词/词汇数
        self.word_p = None # 存储每个单词的概率分布，是一个列表/向量

        # 统计词频，即每个单词出现了几次，即 P(word_i)
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        # （不重复的）单词/词汇总数
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        # 初始化 word_p，然后单词 ID 索引处存储对应单词的词频 P(word_i)
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power) # 求 P(word_i)^power
        self.word_p /= np.sum(self.word_p) # 求 \sum P(word_i)^power（此处使用了广播机制）

    '''
    从小批量（mini-batch）样本中随机取负例样本
    参数说明：
      - target：指定单词 ID（一个列表，即 mini-batch）为正例，以其他单词 ID 作为负例采样
      - 返回负例样本
    '''
    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32) # 初始化

            for i in range(batch_size): # 小批量样本中随机取样
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                '''
                np.random.choice() 参数说明：
                  - size：输出数组的形状。例如，size=3将返回一个包含3个随机选择元素的数组。
                  - replace：是否允许重复选择。默认为True，即允许重复选择；如果设置为False，则进行无放回抽样。
                  - p：与输入数组形状相同的概率数组。它指定了从每个元素中选择的概率。
                '''
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # 在用 GPU(cupy) 计算时，优先速度
            # 有时目标词存在于负例中
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample

# ============================= 层的实现 =============================
# 以后实现的层都有 params（eg.权重和偏置）和 grads 实例变量，由于可能有多个参数，所以均使用列表存储
# 所有的层都有 forward() 和 backward() 方法

'''
二分类&损失层
'''
class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)

        # np.c_：按行连接两个矩阵（左右拼接）
        # np.r_：按列连接两个矩阵（上下拼接）
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
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
        out = W[idx]
        return out

    '''
    # x[0] 和 x[2] 都加 3
    np.add.at(x, [0,2], 3) 

    # 在 x[1,1]、x[3,2]、x[0,1] 都加1
    idx = np.array([1, 3, 0])
    idy = np.array([1, 2, 1])
    np.add.at(x, (idx , idy), 1)
    '''
    def backward(self, dout):
        dW = self.grads
        dW[...] = 0 # 初始化为零矩阵

        np.add.at(dW, self.idx, dout)
        return None

'''
Embedding Dot 层
'''
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)  # 逐元素相乘得到一个新矩阵，然后逐行求和得到一个列表

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh

'''
负采样层
生成 1 个正例层和 sample_size 个负例层
'''
class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size) # 采样器的参数配置

        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)] # 列表中，[0] 为正例所用，其他的都是负例
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)] # 列表中，[0] 为正例所用，其他的都是负例

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0] # 小批量样本个数
        negative_sample = self.sampler.get_negative_sample(target) # 对小批量样本负例采样

        # 正例的正向传播
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32) # 生成正例对应的正确解标签（显然全部都为 1）
        loss = self.loss_layers[0].forward(score, correct_label)

        # 负例的正向传播
        negative_label = np.zeros(batch_size, dtype=np.int32) # 生成负例对应的错确解标签（显然全部都为 0）
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[i + 1].forward(h, negative_target)
            loss += self.loss_layers[i + 1].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh

# ============================= CBOW 模型定义 =============================
class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        self.in_layers = []
        for i in range(window_size * 2):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None

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
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    # 读入训练集数据
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    # 生成上下文与目标词
    contexts, target = create_contexts_target(corpus, window_size)

    # 生成模型
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = SGD()
    trainer = Trainer(model, optimizer)

    # 开始学习
    trainer.fit(contexts, target, max_epoch, batch_size)
    # trainer.plot()

    # 保存训练好的参数
    word_vecs = model.word_vecs
    params = {}
    params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_to_id'] = word_to_id
    params['id_to_word'] = id_to_word
    pkl_file = 'cbow_params.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(params, f, -1)
