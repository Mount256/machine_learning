import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier

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

    adaboost_classifier = AdaBoostClassifier(random_state=0, n_estimators=6)
    adaboost_classifier.fit(features, labels)
    draw_scatter(features, labels)
    draw_boundary(adaboost_classifier, axes=[0, 11, 0, 11])
    plt.savefig("./adaboost/email_adaboost.png")

    cnt = 0
    for e in adaboost_classifier.estimators_:
        cnt += 1
        plt.clf()
        draw_scatter(features, labels)
        draw_boundary(e, axes=[0, 11, 0, 11])
        plt.savefig("./adaboost/email_adaboost_sub" + str(cnt) + ".png")
