import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.pylab as pylab
import seaborn as sns
sns.set()
from cycler import cycler

def use_hds_plot_style(fc=True):
    if fc:
        prop_cycle = cycler(
            color     = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3']*3,
            linestyle = ['-']*8 + ['--']*8 + [':']*8,
            marker    = ['o', 's',  '^', 'v', '<', '>', 'D', ',']*3
        )
    else:
        prop_cycle = cycler(
            color     = ['k', '#777777', '#aaaaaa', '#cccccc']*3,
            linestyle = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.'],
            marker    = ['o', 's',  '^', 'v', '<', '>', 'D', 'o',  's', '^', 'v', '<']
        )

    plt.rc('axes', prop_cycle=prop_cycle, facecolor='#f3f3f3', edgecolor='#f3f3f3',
                   grid=True)
    plt.rc('figure', dpi=200, figsize=(4,3), autolayout=True)
    plt.rc('font', family=['Times', 'Times New Roman', 'serif'])
    plt.rc('grid', color='white', linewidth=1)
    plt.rc('image', origin='lower', cmap='inferno')
    plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white', fancybox=False,
                     numpoints=1, scatterpoints=1, shadow=False)
    plt.rc('lines', markersize=np.sqrt(20))
    plt.rc('ps', papersize='A4')
    plt.rc('savefig', dpi=300)
    plt.rc('scatter', marker='o', edgecolors='face')
    plt.rc('text', usetex=True)
    plt.rc('xtick', bottom=True, top=False)
    plt.rc('xtick.major', size=0)
    plt.rc('ytick', left=True, right=False)
    plt.rc('ytick.major', size=0)
    

use_hds_plot_style(True)

for i in range(24):
    plt.plot(np.arange(4), [i]*4, label='ding')
plt.title('Auto cycled style (full colour)')
plt.ylabel(r'Style \#${i}$')
plt.xlabel(r'log$(\mathcal{L})_{BF}$')
plt.savefig("styles_full_colour.png")


use_hds_plot_style(False)

plt.clf()
for i in range(12):
    plt.plot(np.arange(4), [i]*4, label='ding')
plt.title('Auto cycled style (bw)')
plt.ylabel(r'Style \#${i}$')
plt.xlabel(r'log$(\mathcal{L})_{BF}$')
plt.savefig("styles_bw.png")
