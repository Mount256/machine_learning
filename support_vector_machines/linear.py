import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 绘制散点图
def draw_scatter(features, labels):
    F = np.array(features)
    L = np.array(labels)
    label0_num = np.sum(L==0)
    label1_num = np.sum(L==1)
    F0 = np.zeros((label0_num, 2))
    F1 = np.zeros((label1_num, 2))

    j0 = 0
    j1 = 0
    for i in range(len(L)):
        if L[i] == 0:
            F0[j0] = F[i]
            j0 += 1
        else:
            F1[j1] = F[i]
            j1 += 1

    plt.scatter(F1[:, 0], F1[:, 1], c='r', marker='s', label="y=1")
    plt.scatter(F0[:, 0], F0[:, 1], c='b', marker='^', label="y=0")
    plt.legend(loc='lower right')

# 绘制决策边界
def draw_boundary(clf, axes):
    x1 = np.linspace(axes[0], axes[1], 100)
    x2 = np.linspace(axes[2], axes[3], 100)
    xx, yy = np.meshgrid(x1, x2)
    x_new = np.c_[xx.ravel(), yy.ravel()]
    y_pred = clf.predict(x_new).reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, alpha=0.4, cmap=plt.cm.brg)

# ============================= 主程序 main ===================================
if __name__ == '__main__':
    linear_data = pd.read_csv('linear.csv')
    features = np.array(linear_data[['x_1', 'x_2']])
    labels = np.array(linear_data['y'])
    draw_scatter(features, labels)
    plt.savefig("./linear/linear_data.jpg")

    svm = SVC(kernel='linear')
    svm.fit(features, labels)
    print("[Default] Accuracy=", svm.score(features, labels))
    plt.cla()
    draw_scatter(features, labels)
    draw_boundary(svm, axes=[-3.5, 3.5, -3.5, 3.5])
    plt.savefig("./linear/linear_default.jpg")

    svm_c_001 = SVC(kernel='linear', C=0.01)
    svm_c_001.fit(features, labels)
    print("[C=0.01] Accuracy=", svm_c_001.score(features, labels))
    plt.cla()
    draw_scatter(features, labels)
    draw_boundary(svm_c_001, axes=[-3.5, 3.5, -3.5, 3.5])
    plt.savefig("./linear/linear_c_001.jpg")

    svm_c_100 = SVC(kernel='linear', C=100)
    svm_c_100.fit(features, labels)
    print("[C=100] Accuracy=", svm_c_100.score(features, labels))
    plt.cla()
    draw_scatter(features, labels)
    draw_boundary(svm_c_100, axes=[-3.5, 3.5, -3.5, 3.5])
    plt.savefig("./linear/linear_c_100.jpg")
