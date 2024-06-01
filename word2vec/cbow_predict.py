import sys
sys.path.append('..')
import numpy as np

# ============================= MatMul =============================
# 以后实现的层都有 params（eg.权重和偏置）和 grads 实例变量，由于可能有多个参数，所以均使用列表存储
# 所有的层都有 forward() 和 backward() 方法
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

# ============================= 主程序 main ===================================

if __name__ == '__main__':
    # 上下文数据
    c0 = np.array([1, 0, 0, 0, 0, 0, 0])
    c1 = np.array([0, 0, 0, 1, 0, 0, 0])

    # 初始权重值
    W_in = np.random.randn(7, 3) # 高斯分布
    W_out = np.random.randn(3, 7)

    in_layer0 = MatMul(W_in)
    in_layer1 = MatMul(W_in)
    out_layer = MatMul(W_out)

    h0 = in_layer0.forward(c0)
    h1 = in_layer1.forward(c1)
    h = 0.5 * (h0 + h1)
    s = out_layer.forward(h)

    print(s)
