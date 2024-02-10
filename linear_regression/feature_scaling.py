import numpy as np
import matplotlib.pyplot as plt

# 均值归一化
def mean_normalize_features(X):
    mu = np.mean(X, axis=0) # 计算平均值，矩阵可指定计算行(axis=1)或列(axis=0，此处即特征值)
    X_mean = (X - mu) / (np.max(X, axis=0) - np.min(X, axis=0))
    return X_mean

# z-score 归一化
def zscore_normalize_features(X):
    mu = np.mean(X, axis=0) # 计算平均值，矩阵可指定计算行(axis=1)或列(axis=0，此处即特征值)
    sigma = np.std(X, axis=0) # 计算标准差，矩阵可指定计算行(axis=1)或列(axis=0，此处即特征值)
    X_zscore = (X - sigma) / mu
    return X_zscore

# 绘制两个散点子图
def draw_scatter(X_train, col1, X_norm, col2, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    fig.suptitle("distribution of features before, during, after normalization")

    ax[0].scatter(X_train[:, col1], X_train[:, col2])
    ax[0].set_xlabel("X_train feature 1")
    ax[0].set_ylabel("X_train feature 2")
    ax[0].set_title("unnormalized")

    ax[1].scatter(X_norm[:, col1], X_norm[:, col2])
    ax[1].set_xlabel("X_norm feature 1")
    ax[1].set_ylabel("X_norm feature 2")
    ax[1].set_title(title)

# X[:,0] 二维数组第0列
# X[:,1] 第1列
# X[0,:] 第0行
# X[3,:] 第3行
# X[1:4,:] 第0,1,2行

# 从这里开始执行
if __name__ == '__main__':
    # 训练集样本
    data = np.loadtxt("./data.txt", delimiter=',', skiprows=1)
    X_train = data[:, :4]
    y_train = data[:, 4]

    X_mean = mean_normalize_features(X_train)
    draw_scatter(X_train, 0, X_mean, X_mean.shape[0]-1, "mean normalize")
    plt.show()

    X_zscore = zscore_normalize_features(X_train)
    draw_scatter(X_train, 0, X_zscore, X_zscore.shape[0]-1, "z-score normalize")
    plt.show()
