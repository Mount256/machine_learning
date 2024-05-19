import torch

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]) # 2*3 矩阵

print(y_hat[[0, 1], y]) # 等价于语句 y_hat[[0, 1], [0, 2]]
# 等价于输出 y_hat[0][0] 和 y_hat[1][2] 的值

data = torch.randint(0, 10, [4, 5])  # 四行五列的二维张量
print(data)
print(data[2])  # 获取第三行数据，返回一维张量
print(data[:, 1])  # 获取第二列数据，返回一维张量
print(data[1, 2])  # 获取第二行的第三列数据，返回零维张量
print(data[1][2])  # 同上

