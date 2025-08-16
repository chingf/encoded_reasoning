from dataclasses import dataclass
import os
import numpy as np
import seaborn as sns

sns.set(
        font_scale=10/12.,
        palette='colorblind',
        rc={'axes.axisbelow': True,
            'axes.edgecolor': 'black',  # Changed from lightgrey to black
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'black',  # Changed from dimgrey to black
            'axes.spines.right': True,   # Changed to True to show all spines
            'axes.spines.top': True,     # Changed to True to show all spines
            'text.color': 'black',       # Changed from dimgrey to black

            'lines.solid_capstyle': 'round',
            'lines.linewidth': 1,
            'legend.facecolor': 'white',
            'legend.framealpha': 0.8,

            'xtick.bottom': True,
            'xtick.color': 'black',      # Changed from dimgrey to black
            'xtick.direction': 'out',

            'ytick.color': 'black',      # Changed from dimgrey to black
            'ytick.direction': 'out',
            'ytick.left': True,

            'xtick.major.size': 2,
            'xtick.major.width': .5,
            'xtick.minor.size': 1,
            'xtick.minor.width': .5,

            'ytick.major.size': 2,
            'ytick.major.width': .5,
            'ytick.minor.size': 1,
            'ytick.minor.width': .5})

#import matplotlib.pyplot as plt
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] + plt.rcParams['font.sans-serif']
