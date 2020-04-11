import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.pylab as pylab
import seaborn as sns
sns.set()

x = np.power(10, [-4., -3., -2., -1.])
y = -1*np.array([0.966, 0.943, 0.9855, 0.994])
title = 'Working points = {}'.format(1000)
xlabel = ''
ylabel = 'log$(\mathcal{L})_{BF}$'
xscale_log = False
yscale_log = False
figsize = (4, 3)
dpi = 150

# Fonts
rc('font',**{'family':'serif','serif':['Times New Roman', 'Times', 'serif']})

def get_range(x):
    minimum, maximum = np.amin(x),  np.amax(x)
    r = np.abs(maximum - minimum)
    return ([minimum, maximum], r)

plt.figure(figsize=figsize, dpi=dpi)
plt.scatter(x, y)

xlim, xr = get_range(x)
if xscale_log:
    plt.xscale('log')
    xlim, xr = np.log10(xlim), np.abs(np.log10(xr))
    xlim = np.power(10, [xlim[0]-0.2*xr, xlim[1]+0.2*xr])
plt.xlim(xlim)

if yscale_log:
    plt.yscale('log')

plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.tight_layout()
plt.show()
