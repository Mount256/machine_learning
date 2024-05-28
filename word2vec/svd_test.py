import numpy as np
from sklearn.utils.extmath import randomized_svd

a = np.array([[1, 2, 3, 5],
              [3, 4, 5, 6],
              [7, 8, 9, 10]])

U, s, Vh = randomized_svd(a, n_components=2, random_state=0)

print(U)
print(s)
print(Vh)
