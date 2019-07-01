"""
This script shows how test functions from the functions module can be used.
"""
import high_dimensional_sampling as hds
import numpy as np
import matplotlib.pyplot as plt

f = hds.functions.Cosine()
x = np.linspace(-6, 6, 1000).reshape(-1, 1)
y = f(x)
yprime = f(x, True)

plt.plot(x, y, label="Cosine")
plt.plot(x, yprime, label="-Sine (derivative)")
plt.grid()
plt.legend()
plt.show()
