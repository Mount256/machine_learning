# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
import pickle

# CBOW 模型的评价

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

# ============================= 主程序 main ===================================

if __name__ == '__main__':
    pkl_file = "cbow_params.pkl"

    with open(pkl_file, 'rb') as f:
        params = pickle.load(f)
        word_vecs = params['word_vecs']
        word_to_id = params['word_to_id']
        id_to_word = params['id_to_word']

    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
