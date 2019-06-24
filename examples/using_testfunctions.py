import sys
import high_dimensional_sampling as hds
import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, '/Users/bstienen/GitHub/high-dimensional-sampling/high_dimensional_sampling')

N = 2000
x = np.linspace(-1, 1, N)*5
x = x.reshape(-1, 1)

z = hds.functions.Sphere()
y = z(x)
yprime = z(x, True)

plt.plot(x, y, label="normal")
plt.plot(x, yprime, label="derivative")
plt.legend()
plt.grid()
plt.show()
