# 集成方法的种类

- 集成方法有两种：bagging 和 boosting
  - bagging（bootstrap）集成：在数据的随机子集上构建连续的弱学习器，并将这些弱学习器组合成一个强学习器，它基于多数投票进行预测
  - boosting 集成： 构建一系列学习器，每个学习器专注于前一个学习器的弱点。boosting 将这些学习器组合成一个强大的分类器，该分类器根据学习器的加权投票进行预测

# 【bagging】随机森林

待补充

# 【boosting】AdaBoost

若一个事件发生的概率为 $x$，则对数赔率为

$$
log-odds(x) = \ln \frac{x}{1-x}
$$
