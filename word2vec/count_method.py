import sys
sys.path.append('..')
import numpy as np

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
共现矩阵（co_matrix）的生成
输入：
    - corpus：语料库
    - vocab_size：
    - window_size：窗口大小（左右各多少个单词）
输出：
    - co_matrix：共现矩阵
'''
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus) # 获取语料库的大小
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i # 左边单词索引
            right_idx = idx + i # 右边单词索引

            if left_idx >= 0:
                left_word_id = corpus[left_idx] # 获取左边单词的 id 号
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]  # 获取右边单词的 id 号
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

'''
向量之间的余弦相似度
eps 的作用：当 x 或 y 为零向量时，避免出现除数为零的问题
'''
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

'''
查询指定单词与其他单词的相似度，按相似度降序输出这些单词
输入：
    - query：待查询单词
    - word_to_id：
    - id_to_word：
    - co_matrix：共现矩阵
    - top：输出相似度前几位的单词
输出：无
'''
def most_similar(query, word_to_id, id_to_word, co_matrix, top=5):
    if query not in word_to_id: # 若语料库中没有待查词
        print(f"{query} is not found")
        return

    print("\n[query]" + query)
    query_id = word_to_id[query]
    query_vec = co_matrix[query_id]

    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(co_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:  # 待查词自身的相似度就不用输出了
            continue
        print(f"{id_to_word[i]}: {similarity[i]}")

        count += 1
        if count >= top:
            return

'''
将共现矩阵转化为 PPMI 矩阵
PPMI(x, y) = max(0, PMI(x, y))
PMI(x, y) = log_2 \frac{C(x, y) \dot N} {C(x) C(y)}
输入：
    - C：共现矩阵
    - verbose：是否输出运行情况的标志
    - eps：防止计算 log 时出现无穷大的情况
输出：
    - M：PPMI 矩阵

对于二维矩阵求和，
当 axis=0 时，求和就是矩阵投影到 x 轴的结果
当 axis=1 时，求和就是矩阵投影到 y 轴的结果
'''
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32) # PPMI 矩阵
    N = np.sum(C) # 矩阵中所有元素相加，是一个标量
    S = np.sum(C, axis=0) # 矩阵每列的所有元素相加，是一个向量
    total = C.shape[0] * C.shape[1]
    cnt = 0

    print(N)
    print(S)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print('%.1lf%% done' % (100*cnt/total))

    return M

# ============================= 主程序 main ===================================

if __name__ == '__main__':
    text1 = "You say goodbye and I say hello."
    text2 = "I think smiling is as important as sunshine. Smiling is like sunshine because it can make people happy and have a good day. If you aren’t happy, you can smile, and then you will feel happy. Someone may say, But I don’t feel happy. Then I would say, Please smile as you do when you are happy or play with your friends happily. You will really be happy again."
    corpus, word_to_id, id_to_word = preprocess(text1)

    print(corpus)
    print(word_to_id)
    print(id_to_word)

    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size, window_size=1)
    #print(C)

    most_similar('you', word_to_id, id_to_word, C)

    W = ppmi(C)
    print(W)
