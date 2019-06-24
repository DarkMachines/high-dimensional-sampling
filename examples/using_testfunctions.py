import sys
sys.path.insert(0, '/Users/bstienen/GitHub/high-dimensional-sampling/high_dimensional_sampling')

import testfunctions
import matplotlib.pyplot as plt
import numpy as np

N = 2000
x = np.linspace(-1, 1, N)*5
x = x.reshape(-1,1)

z = testfunctions.Sine()
y = z(x)
yprime = z(x, True)

plt.plot(x, y, label="normal")
plt.plot(x, yprime, label="derivative")
plt.legend()
plt.grid()
plt.show()

w = testfunctions.TestFunction()
w([[0,1,2]])