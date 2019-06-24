import high_dimensional_sampling as hds
import numpy as np
import matplotlib.pyplot as plt


f = hds.functions.ThreeHumpCamel()
x = np.random.rand(10000, 2)*4-2
y = f(x)
y = np.log10(y)

plt.scatter(x[:,0], x[:,1], c=y)
plt.show()