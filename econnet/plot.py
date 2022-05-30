import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set plotting font sizes and properties
TINY_SIZE = 12
SMALL_SIZE = 28
MEDIUM_SIZE = 28
BIGGER_SIZE = 20
MARKER_SIZE = 6
LINE_SIZE = 4

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=TINY_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("lines", markersize=MARKER_SIZE)  # marker size
plt.rc("lines", linewidth=LINE_SIZE)  # line width

mpl.rcParams["figure.dpi"] = 180 # sets the image quality

# Height and width per row and column of subplots
FIG_HEIGHT = 18
FIG_WIDTH = 25
fig_fcn = lambda kwargs: plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), **kwargs)
color_list = sns.color_palette("Paired")


def barplot(series, savefig=False, title="", xlab="", fig_path="figs/"): 
    """
    Seaborn barplot of Pandas series
    :param: series - pandas series
    :param: savefig - boolean - save fig or not
    :param: title, xlab - strings for plot
    :param: fig_path - string for figure folder/path
    :return: matplotlib figure
    """
    fig_fcn({"num": None, "dpi": 80, "facecolor": "w"})
    fig = sns.barplot(series.values, series.index, color="deepskyblue")
    plt.xlabel(xlab)
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_path + f"{title}.png", dpi=350)

    return fig

def corrplot(corr, savefig=False, title="", xlab="", fig_path="figs/"):
    """
    Seaborn corrplot of pandas dataframe
    :param: corr - pandas dataframe consisting of correlations
    :param: savefig - boolean - save fig or not
    :param: title, xlab - strings for plot
    :param: fig_path - string for figure folder/path
    :return: matplotlib figure
    """
    fig_fcn({"num": None, "dpi": 80, "facecolor": "w"})
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    fig = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.xlabel(xlab)
    if savefig:
        plt.savefig(fig_path + f"{title}.png", dpi=350)

    return fig