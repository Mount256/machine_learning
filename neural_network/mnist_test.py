# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 从这里开始执行
if __name__ == '__main__':
    # 展开为一维数组，且不使用归一化（详细参数说明见/dataset/mnist.py）
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    img = x_train[0]
    label = t_train[0]
    print(label)  # 5

    print(img.shape)  # (784,)
    img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
    print(img.shape)  # (28, 28)
    print(img) # 打印图像的灰度值

    img_show(img)
