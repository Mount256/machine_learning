import pickle

def pickle_test_1():
    a = {'name': 'Tom', 'age': 18}

    with open('pickle_test.pkl', 'wb') as file1:
        pickle.dump(a, file1)

    with open('pickle_test.pkl', 'rb') as file2:
        b = pickle.load(file2)
    print(type(b))
    print(b)

def pickle_test_2():
    with open('sample_weight.pkl', 'rb') as f:
        weight = pickle.load(f)
    return weight

# 从这里开始执行
if __name__ == '__main__':
    # pickle_test_1()
    w = pickle_test_2()
    print(type(w))
    print(w['W1'])
    print(w['b1'])

