import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


def use_hds_plot_style(full_colour=True):
    """
    Applies the HDS plot style to any of the plots made with matplotlib.

    Args:
        full_colour: `bool` indicating if the plots should be in
            full-colour (`True`) or in black and white (`False`).
    """
    if full_colour:
        prop_cycle = cycler(color=[
            '#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F',
            '#E5C494', '#B3B3B3'
        ] * 3,
                            linestyle=['-'] * 8 + ['--'] * 8 + [':'] * 8,
                            marker=['o', 's', '^', 'v', '<', '>', 'D', ','] *
                            3)
    else:
        prop_cycle = cycler(color=['k', '#777777', '#aaaaaa', '#cccccc'] * 3,
                            linestyle=[
                                '-', '--', ':', '-.', '-', '--', ':', '-.',
                                '-', '--', ':', '-.'
                            ],
                            marker=[
                                'o', 's', '^', 'v', '<', '>', 'D', 'o', 's',
                                '^', 'v', '<'
                            ])

    plt.rc('axes',
           prop_cycle=prop_cycle,
           facecolor='#f3f3f3',
           edgecolor='#f3f3f3',
           grid=True)
    plt.rc('figure', dpi=200, figsize=(4, 3), autolayout=True)
    plt.rc('font', family=['Times', 'Times New Roman', 'serif'])
    plt.rc('grid', color='white', linewidth=1)
    plt.rc('image', origin='lower', cmap='inferno')
    plt.rc('legend',
           facecolor='white',
           framealpha=1,
           edgecolor='white',
           fancybox=False,
           numpoints=1,
           scatterpoints=1,
           shadow=False)
    plt.rc('lines', markersize=np.sqrt(20))
    plt.rc('ps', papersize='A4')
    plt.rc('savefig', dpi=300)
    plt.rc('scatter', marker='o', edgecolors='face')
    plt.rc('text', usetex=True)
    plt.rc('xtick', bottom=True, top=False)
    plt.rc('xtick.major', size=0)
    plt.rc('ytick', left=True, right=False)
    plt.rc('ytick.major', size=0)
