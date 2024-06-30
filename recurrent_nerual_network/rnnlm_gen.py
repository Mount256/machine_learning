# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from dataset import ptb

# ============================= 可选用不同的RNNLM模型生成文本 =============================
BetterRNNLM = False

if BetterRNNLM:
    from better_rnnlm import RNNLM
else:
    from rnnlm import RNNLM

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


# ============================= 生成文本的类（继承RNNLM类） ===================================
'''
参数说明：
- start_id：第一个单词ID
- skip_ids：指定不被采样（或被舍弃）的单词ID列表，如标点符号、阿拉伯数字等预处理过的单词
- sample_size：要采样的单词数，即生成文本的单词数
'''
class RnnlmGen(RNNLM):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id] # 记录文本，注意记录的不是单词本身而是单词 ID

        x = start_id # 文本开头第一个单词
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1) # 由于 RNNLM 的输入必须为矩阵，所以要预先将 x 转化为矩阵形式
            score = self.predict(x) # 基于输入的单词 x，计算 x 后一个单词中各个单词的得分，输出大小为(1, 1, vocab_size)
            p = softmax(score.flatten()) # 将得分正规化，得出每个单词的概率分布

            # len(p)：单词 ID 的采样范围，size=1：只采样一个单词，p=p：根据概率分布 p 采样单词 ID
            sampled = np.random.choice(len(p), size=1, p=p)

            # 如果采样到的单词不属于舍弃列表，则会被写入文本中
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids


# ============================= 主程序 main ===================================
if __name__ == '__main__':
    corpus, word_to_id, id_to_word = ptb.load_data('train') # 读入训练集数据
    vocab_size = len(word_to_id) # 词汇数
    corpus_size = len(corpus) # 语料库的单词数

    model = RnnlmGen()
    # 如果不使用训练好的权重数据，则生成的是杂乱无章的文本，因为网络权重值都是随机的
    model.load_params('RNNLM.pkl')  # 取出训练好的权重数据

    # 指定一个单词，生成文本
    start_word = 'you'
    start_id = word_to_id[start_word]
    skip_words = ['N', '<unk>', '$'] # PTB语料库中已经处理的标点符号和数字
    skip_ids = [word_to_id[w] for w in skip_words]

    word_ids = model.generate(start_id, skip_ids) # 这里得到的是一堆单词 ID
    txt = ' '.join(id_to_word[i] for i in word_ids) # 将单词 ID 转化为单词本身，顺便加空格间隔
    txt = txt.replace(' <eos>', '.\n')
    print(txt)

    model.reset_state()

    # 指定一串单词，生成文本
    start_words = 'the meaning of life is'
    start_ids = [word_to_id[w] for w in start_words.split(' ')]

    for x in start_ids[:-1]: # 除了最后一个单词外，把前面的单词一一输入到网络中
        x = np.array(x).reshape(1, 1)
        model.predict(x) # 输出的是 xs(1, 1, vocab_size)

    word_ids = model.generate(start_ids[-1], skip_ids)
    word_ids = start_ids[:-1] + word_ids
    txt = ' '.join([id_to_word[i] for i in word_ids])
    txt = txt.replace(' <eos>', '.\n')
    print('-' * 50)
    print(txt)
