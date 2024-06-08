import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


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
    dataset = pd.DataFrame({
        'x_0': [7, 3, 2, 1, 2, 4, 1, 8, 6, 7, 8, 9],
        'x_1': [1, 2, 3, 5, 6, 7, 9, 10, 5, 8, 4, 6],
        'y': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})
    features = dataset[['x_0', 'x_1']]
    labels = dataset['y']

    decision_tree = DecisionTreeClassifier(criterion='gini')
    decision_tree.fit(features, labels)
    decision_tree.score(features, labels)
    tree.plot_tree(decision_tree, rounded=True, feature_names=['x_0', 'x_1'], class_names=['0', '1'])
    plt.savefig("./graphic_example/graphic_example_gini.png")
    plt.clf() # 清空画布
    draw_scatter(features, labels)
    draw_boundary(decision_tree, axes=[0, 11, 0, 11])
    plt.savefig("./graphic_example/graphic_example_gini(boundary).png")

    decision_tree_entropy = DecisionTreeClassifier(criterion='entropy')
    decision_tree_entropy.fit(features, labels)
    decision_tree_entropy.score(features, labels)
    tree.plot_tree(decision_tree_entropy, rounded=True, feature_names=['x_0', 'x_1'], class_names=['0', '1'])
    plt.savefig("./graphic_example/graphic_example_entropy.png")
    plt.clf()
    draw_scatter(features, labels)
    draw_boundary(decision_tree_entropy, axes=[0, 11, 0, 11])
    plt.savefig("./graphic_example/graphic_example_entropy(boundary).png")

    decision_tree_depth_1 = DecisionTreeClassifier(criterion='gini', max_depth=1)
    decision_tree_depth_1.fit(features, labels)
    decision_tree_depth_1.score(features, labels)
    tree.plot_tree(decision_tree_depth_1, rounded=True, feature_names=['x_0', 'x_1'], class_names=['0', '1'])
    plt.savefig("./graphic_example/graphic_example_depth_1.png")
    plt.clf()
    draw_scatter(features, labels)
    draw_boundary(decision_tree_depth_1, axes=[0, 11, 0, 11])
    plt.savefig("./graphic_example/graphic_example_depth_1(boundary).png")
