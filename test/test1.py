import matplotlib.pyplot as plt
from random import randint

linear = list(range(1, 21)) # linear列表从1平滑增加到20
wiggly = list(num + randint(-1, 1) for num in linear) # wiggly列表与linear列表类似但有随机的轻微抖动

fig, plots = plt.subplots(nrows=1, ncols=3) # 布置三个子图

# 设置三个子图的x轴与y轴的刻度
ticks = list(range(0, 21, 5))
for plot in plots:
    plot.set_xticks(ticks)
    plot.set_yticks(ticks)

# 绘制三个子图的图形
plots[0].scatter(linear, wiggly)
plots[1].plot(linear, wiggly)
plots[2].plot(linear, wiggly, 'o-')

# 展示图
plt.show()
