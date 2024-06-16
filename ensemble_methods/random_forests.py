import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# ============================= 数据集 dataset ===================================
emails = np.array([
    [7,8,1],
    [3,2,0],
    [8,4,1],
    [2,6,0],
    [6,5,1],
    [9,6,1],
    [8,5,0],
    [7,1,0],
    [1,9,1],
    [4,7,0],
    [1,3,0],
    [3,10,1],
    [2,2,1],
    [9,3,0],
    [5,3,0],
    [10,1,0],
    [5,9,1],
    [10,8,1],
])

'''
特征：“lottery”和“sale”这两个词在邮件中的出现次数
标签：该邮件是否为垃圾邮件（spam），是为 1，否为 0
'''
spam_dataset = pd.DataFrame(data=emails, columns=["Lottery", "Sale", "Spam"])

# ============================= 公用函数 common ===================================
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

    plt.scatter(F1[:, 0], F1[:, 1], c='r', marker='s', label="spam")
    plt.scatter(F0[:, 0], F0[:, 1], c='b', marker='^', label="ham")
    plt.xlabel('Lottery')
    plt.ylabel('Sale')
    plt.legend(loc='upper right')

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
    features = spam_dataset[['Lottery', 'Sale']]
    labels = spam_dataset['Spam']

    # 决策树模型训练（过拟合）
    decision_tree_classifier = DecisionTreeClassifier(random_state=0)
    decision_tree_classifier.fit(features, labels)
    tree.plot_tree(decision_tree_classifier, rounded=True, feature_names=['Lottery', 'Sale'], class_names=['Ham', 'Spam'])
    plt.savefig("./random_forests/email_decision_tree_1.png")
    plt.clf()
    draw_scatter(features, labels)
    draw_boundary(decision_tree_classifier, axes=[0, 11, 0, 11])
    plt.savefig("./random_forests/email_decision_tree_2.png")

    # 随机森林（手动拟合，数据集分为3个子集）
    first_batch = spam_dataset.loc[[0, 1, 2, 3, 4, 5]]
    features1 = first_batch[['Lottery', 'Sale']]
    labels1 = first_batch['Spam']
    second_batch = spam_dataset.loc[[6, 7, 8, 9, 10, 11]]
    features2 = second_batch[['Lottery', 'Sale']]
    labels2 = second_batch['Spam']
    third_batch = spam_dataset.loc[[12, 13, 14, 15, 16, 17]]
    features3 = third_batch[['Lottery', 'Sale']]
    labels3 = third_batch['Spam']

    dt1 = DecisionTreeClassifier(random_state=0, max_depth=1)
    dt1.fit(features1, labels1)
    tree.plot_tree(dt1, rounded=True)
    plt.savefig("./random_forests/email_decision_tree_sub11.png")
    plt.clf()
    draw_scatter(features1, labels1)
    draw_boundary(dt1, axes=[0, 11, 0, 11])
    plt.savefig("./random_forests/email_decision_tree_sub12.png")

    dt2 = DecisionTreeClassifier(random_state=0, max_depth=1)
    dt2.fit(features2, labels2)
    tree.plot_tree(dt2, rounded=True)
    plt.savefig("./random_forests/email_decision_tree_sub21.png")
    plt.clf()
    draw_scatter(features2, labels2)
    draw_boundary(dt2, axes=[0, 11, 0, 11])
    plt.savefig("./random_forests/email_decision_tree_sub22.png")

    dt3 = DecisionTreeClassifier(random_state=0, max_depth=1)
    dt3.fit(features3, labels3)
    tree.plot_tree(dt3, rounded=True)
    plt.savefig("./random_forests/email_decision_tree_sub31.png")
    plt.clf()
    draw_scatter(features3, labels3)
    draw_boundary(dt3, axes=[0, 11, 0, 11])
    plt.savefig("./random_forests/email_decision_tree_sub32.png")

    # 使用API训练随机森林
    random_forest_classifier = RandomForestClassifier(random_state=0, n_estimators=5, max_depth=1)
    random_forest_classifier.fit(features, labels)

    cnt = 0
    for dt in random_forest_classifier.estimators_:
        cnt += 1
        tree.plot_tree(dt, rounded=True)
        plt.savefig("./random_forests/email_random_forests_sub" + str(cnt) + "1.png")
        plt.clf()
        draw_scatter(features, labels)
        draw_boundary(dt, axes=[0, 11, 0, 11])
        plt.savefig("./random_forests/email_random_forests_sub" + str(cnt) + "2.png")

    plt.clf()
    draw_scatter(features, labels)
    draw_boundary(random_forest_classifier, axes=[0, 11, 0, 11])
    plt.savefig("./random_forests/email_random_forests.png")
