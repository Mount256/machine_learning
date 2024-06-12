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
    circular_data = pd.read_csv('two_circles.csv')
    features = np.array(circular_data[['x_1', 'x_2']])
    labels = np.array(circular_data['y'])
    draw_scatter(features, labels)
    plt.savefig("./two_circles/two_circles.jpg")

    svm_gamma_01 = SVC(kernel='rbf', gamma=0.1)
    svm_gamma_01.fit(features, labels)
    print("[Gamma=0.1] Accuracy=", svm_gamma_01.score(features, labels))
    plt.cla()
    draw_scatter(features, labels)
    draw_boundary(svm_gamma_01, axes=[-3.5, 3.5, -3.5, 3.5])
    plt.savefig("./two_circles/two_circles_gamma_01.jpg")

    svm_gamma_1 = SVC(kernel='rbf', gamma=1)
    svm_gamma_1.fit(features, labels)
    print("[Gamma=1] Accuracy=", svm_gamma_1.score(features, labels))
    plt.cla()
    draw_scatter(features, labels)
    draw_boundary(svm_gamma_1, axes=[-3.5, 3.5, -3.5, 3.5])
    plt.savefig("./two_circles/two_circles_gamma_1.jpg")

    svm_gamma_10 = SVC(kernel='rbf', gamma=10)
    svm_gamma_10.fit(features, labels)
    print("[Gamma=10] Accuracy=", svm_gamma_10.score(features, labels))
    plt.cla()
    draw_scatter(features, labels)
    draw_boundary(svm_gamma_10, axes=[-3.5, 3.5, -3.5, 3.5])
    plt.savefig("./two_circles/two_circles_gamma_10.jpg")

    svm_gamma_100 = SVC(kernel='rbf', gamma=100)
    svm_gamma_100.fit(features, labels)
    print("[Gamma=100] Accuracy=", svm_gamma_100.score(features, labels))
    plt.cla()
    draw_scatter(features, labels)
    draw_boundary(svm_gamma_100, axes=[-3.5, 3.5, -3.5, 3.5])
    plt.savefig("./two_circles/two_circles_gamma_100.jpg")


