import cupy as cp

# 创建形状为 (3, 4)、元素为 1 到 9 的数组
arr1 = cp.random.randint(1, 10, size=(3, 4))
print(arr1)

# 创建形状为 (4, 3)、元素为 1 到 9 的数组
arr2 = cp.random.randint(1, 10, size=(4, 3))
print(arr2)

# 矩阵乘法
print(arr1 @ arr2)

# 创建形状为 (3, 4)、元素为 0 到 1 的数组
arr2 = cp.random.random((3, 4))
print(arr2)

# 创建形状为 (3, 4)、元素为 1 到 9 的数组，但元素是浮点数
arr3 = cp.random.uniform(1, 10, size=(3, 4))
print(arr3)

# 按照正态分布创建形状为 (3, 4) 的数组，均值为 1、标准差为 0.5
arr4 = cp.random.normal(1, 0.5, size=(3, 4))
print(arr4)

