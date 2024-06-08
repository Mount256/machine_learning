import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.cos(X) + np.sin(Y)

plt.contourf(X, Y, Z, levels=10)
plt.colorbar()
plt.show()
