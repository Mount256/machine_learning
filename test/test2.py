# 按要求绘制以下点的散点图、折线图、带标记的折线图
# (0, 0) (3, 5) (6, 2) (9, 8) (14, 10)

import matplotlib.pyplot as plt

x = [0, 3, 6, 9, 14]
y = [0, 5, 2, 8, 10]

fig, plots = plt.subplots(nrows=1, ncols=3) # 布置三个子图

# 设置三个子图的x轴与y轴的刻度
ticks = list(range(0, 15, 5))
for plot in plots:
    plot.set_xticks(ticks)
    plot.set_yticks(ticks)

# 绘制三个子图的图形
plots[0].scatter(x, y)
plots[1].plot(x, y)
plots[2].plot(x, y, 'o-')

# 展示图
plt.show()
