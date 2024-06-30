# coding: utf-8
import sys
sys.path.append('..')
from dataset import sequence

(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt', seed=1984)
char_to_id, id_to_char = sequence.get_vocab()

print(len(char_to_id))

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

print(x_train[1])
print(t_train[1])

print(''.join([id_to_char[c] for c in x_train[1]]))
print(''.join([id_to_char[c] for c in t_train[1]]))
