import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# ============================= 主程序 main ===================================
if __name__ == '__main__':
    app_dataset = pd.DataFrame({
        'Platform': ['iPhone','iPhone','Android','iPhone','Android','Android'],
        'Age': [15, 25, 32, 35, 12, 14],
        'App': ['Atom Count', 'Check Mate Mate', 'Beehive Finder', 'Check Mate Mate', 'Atom Count', 'Atom Count']})
    print(app_dataset)

    app_dataset_one_hot = pd.DataFrame(
        {'Platform_iPhone': [1, 1, 0, 1, 0, 0],
         'Platform_Android': [0, 0, 1, 0, 1, 1],
         'Age_Young': [1, 0, 0, 0, 1, 1],
         'Age_Adult': [0, 1, 1, 1, 0, 0],
         'App_Atom_Count': [1, 0, 0, 0, 1, 1],
         'App_Beehive_Finder': [0, 0, 1, 0, 0, 0],
         'App_Check_Mate_Mate': [0, 1, 0, 1, 0, 0]})

    X = app_dataset_one_hot[['Platform_iPhone', 'Platform_Android', 'Age_Adult', 'Age_Young']]
    y = app_dataset_one_hot[['App_Atom_Count', 'App_Beehive_Finder', 'App_Check_Mate_Mate']]

    '''
    DecisionTreeClassifier() 部分参数值
        criterion = gini/entropy 可以用来选择用基尼指数或者熵来做损失函数。
        splitter = best/random 用来确定每个节点的分裂策略。支持 “最佳” 或者“随机”。
        max_depth = int 用来控制决策树的最大深度，防止模型出现过拟合。
        min_samples_leaf = int 用来设置叶节点上的最少样本数量，用于对树进行修剪。
    '''
    dt = DecisionTreeClassifier(criterion='gini')
    dt.fit(X, y)
    dt.score(X, y)

    # plot_tree() 部分参数值如下：
    # decision_tree：填入训练好的分类树模型
    # max_depth：填入整数类型数据，可以控制树的最大深度
    # feature_names = None：填入特征值的名称，不填默认是x[0],x[1],x[2],取决特征值数量
    # class_names = None：填入目标值的分类名称，不填不显示，如果是None则是x[0],x[1],x[2],取决于目标值的类别数量
    # label = ["all" or "root" or "None"]按照官方的说法是是否显示杂项数据，默认all显示所有，root只在根节点显示，None不显示
    tree.plot_tree(dt, rounded=True,
                   feature_names=['Platform_iPhone', 'Platform_Android', 'Age_Adult', 'Age_Young'],
                   class_names=['App_Atom_Count', 'App_Check_Mate_Mate', 'App_Beehive_Finder'])
    plt.savefig("./app_recommendations/app_recommendations.png")
