from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import copy

'''
【鸢尾花数据集】样本数量150个，每类50个
4个特征：sepal length（萼片长度）、sepal width（萼片宽度）、petal length（花瓣长度）、petal width （花瓣宽度）
3个属性：Setosa、Versicolour、Virginica
'''

# 绘制散点图
def draw_scatter(features, x1_label, x2_label, title, legend_loc):
    plt.scatter(features[0:50, 0], features[0:50, 1], c='r', marker='s', label="Setosa")
    plt.scatter(features[51:100, 0], features[51:100, 1], c='g', marker='^', label="Versicolour")
    plt.scatter(features[101:150, 0], features[101:150, 1], c='b', marker='o', label="Virginica")
    plt.xlabel(x1_label)
    plt.ylabel(x2_label)
    plt.title(title)
    plt.legend(loc=legend_loc)


# 绘制决策边界
def draw_boundary(clf, axes):
    x1 = np.linspace(axes[0], axes[1], 100)
    x2 = np.linspace(axes[2], axes[3], 100)
    xx, yy = np.meshgrid(x1, x2) # 生成 v 网格采样点
    x_new = np.c_[xx.ravel(), yy.ravel()] # ravel()方法将数组维度拉成一维数组
    y_pred = clf.predict(x_new).reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, alpha=0.4, cmap=plt.cm.brg) # 绘制等高线函数


# ============================= 主程序 main ===================================
if __name__ == '__main__':
    # 加载数据
    iris = load_iris()
    #print(iris['DESCR'])

    features = iris['data'][:, 0:2]
    draw_scatter(features, "sepal length (cm)", "sepal width (cm)", "sepal length and width", 'upper right')
    plt.savefig("./iris/iris_sepal.png")
    plt.cla()

    features = iris['data'][:, 2:4]
    draw_scatter(features, "pental length (cm)", "pental width (cm)", "pental length and width", 'upper left')
    plt.savefig("./iris/iris_pental.png")

    # 训练决策树
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None)
    clf = clf.fit(iris.data, iris.target)
    tree.plot_tree(clf, rounded=True, feature_names=iris.feature_names, class_names=iris.target_names)
    r = tree.export_text(clf, feature_names=iris.feature_names, class_names=iris.target_names)
    print("训练好的决策树用文字表示如下：")
    print(r)
    plt.savefig("./iris/iris_gini_depth_None.png")
    plt.cla()
    #draw_boundary(clf, axes=[0, 8.0, 0, 4.5])

    clf = DecisionTreeClassifier(criterion='gini', max_depth=2)
    clf = clf.fit(iris.data, iris.target)
    tree.plot_tree(clf, rounded=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.savefig("./iris/iris_gini_depth_2.png")
    plt.cla()

    clf = DecisionTreeClassifier(criterion='gini', max_depth=4)
    clf = clf.fit(iris.data, iris.target)
    tree.plot_tree(clf, rounded=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.savefig("./iris/iris_gini_depth_4.png")
