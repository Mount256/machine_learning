import sys
sys.path.append('..')
import numpy as np
from trainer import Trainer

'''
skip-gram 模型使用的神经网络的输入是一个单词，要预测（输出）的是上下文单词，
这跟英语的【完形填空】恰好相反
'''

# ============================= 公用函数 =============================
'''
语料库的预处理
'''
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word

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

'''
转换为one-hot表示
    :param corpus: 语料库转化为以单词ID表示的列表（一维或二维的NumPy数组）
    :param vocab_size: 词汇个数（不重复）
    :return: one-hot表示（二维或三维的NumPy数组）
    若输入为 contexts：输出为三维数组，大小为 (len(corpus)-window_size*2, window_size*2, vocab_size)
    若输入为 tearget：输出为二维数组，大小为 (len(corpus)-window_size*2, vocab_size)
'''
def convert_one_hot(corpus, vocab_size):
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

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

# ============================= 层的实现 =============================
# 以后实现的层都有 params（eg.权重和偏置）和 grads 实例变量，由于可能有多个参数，所以均使用列表存储
# 所有的层都有 forward() 和 backward() 方法

'''
Matmul 层
'''
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params # 这里的逗号为了取出权重 W 而不是 [W]
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW  # 深复制，grads[0] = dW 是浅复制
        return dx

'''
分类&损失层
'''
class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax 的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 在监督标签为 one-hot 向量的情况下，转换为正确解标签的索引
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx

# ============================= skip-gram 模型定义 =============================
class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer0 = SoftmaxWithLoss()
        self.loss_layer1 = SoftmaxWithLoss()

        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        l0 = self.loss_layer0.forward(s, contexts[:, 0])
        l1 = self.loss_layer1.forward(s, contexts[:, 1])
        l = l0 + l1
        return l

    def backward(self, dout=1):
        dl0 = self.loss_layer0.backward(dout)
        dl1 = self.loss_layer1.backward(dout)
        ds = dl0 + dl1
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
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
    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000
    text1 = "You say goodbye and I say hello."
    text2 = "I think smiling is as important as sunshine. Smiling is like sunshine because it can make people happy and have a good day. If you aren’t happy, you can smile, and then you will feel happy. Someone may say, But I don’t feel happy. Then I would say, Please smile as you do when you are happy or play with your friends happily. You will really be happy again."

    corpus, word_to_id, id_to_word = preprocess(text2)
    vocab_size = len(word_to_id)

    contexts, target = create_contexts_target(corpus, window_size)
    contexts = convert_one_hot(contexts, vocab_size)
    target = convert_one_hot(target, vocab_size)

    '''
    此 skip-gram 结构如下：
      - 输入层（Matmul）大小：vocab_size * hidden_size，输入向量大小：(len(corpus)-window_size*2) * vocab_size
      - 输出层（Matmul）大小：hidden_size * vocab_size，输出向量大小：(len(corpus)-window_size*2) * vocab_size
    '''
    model = SimpleSkipGram(vocab_size, hidden_size)

    # 优化器选择随机梯度下降算法
    optimizer = SGD()

    # 训练器参数初始化
    trainer = Trainer(model, optimizer)

    '''
    参数说明：
      - contexts：上下文
      - target：目标词（正确解）
      - max_epoch：
      - batch_size：小批量样本大小
    '''
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()

    word_vecs = model.word_vecs
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])
