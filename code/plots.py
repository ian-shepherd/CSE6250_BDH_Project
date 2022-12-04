import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import numpy as np
import plotly

# import pylab as plt
# %matplotlib inline
# from chart_studio import plotly

# plotly.tools.set_credentials_file(username='khahuras', api_key='AlSpmDRMFqgFHQELDmWB')


def show_plot(df, title, ylabel, file_name, PATH_FILES="./images/"):
    # PATH_FILES = "../images/"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.locator_params(axis="x", nbins=22)
    ax.locator_params(axis="y", nbins=12)

    for i in df.columns[1:]:
        plt.plot(df[i], label=i)

    plt.xlabel("Hyperparameters")
    plt.ylabel(ylabel)
    plt.title(title)

    xlabels = df["param"]
    # ylabels = ['',0,0.1,0.2,0.3,0.4,0.5,.6,.7,.8,.9,1.0]

    ax.set_xticklabels(xlabels, rotation="vertical")
    # ax.set_yticklabels(ylabels)

    legend = ax.legend(loc="lower center", shadow=True, fontsize="large")
    # Put a nicer background color on the legend.
    # legend.get_frame().set_facecolor('#00FFCC')

    plt.show()
    fig.savefig(PATH_FILES + file_name, bbox_inches="tight")


def show_diagram(df, title, filename):
    # plt.figure(figsize=(7,7))
    fig, ax = plt.subplots(figsize=(6, 6))
    v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels=("SVM-bag", "LR-bag", "NN-bag"))
    # v.get_patch_by_id('100').set_alpha(1.0)
    # v.get_patch_by_id('100').set_color('white')
    # v.get_label_by_id('100').set_text('Unknown')
    # v.get_label_by_id('A').set_text('SVM-bag')
    c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linewidth=1)  # linestyle='dashed',
    # c[0].set_lw(1.0)
    v.get_label_by_id("001").set_text("402")
    v.get_label_by_id("010").set_text("926")
    v.get_label_by_id("011").set_text("281")
    v.get_label_by_id("111").set_text("180")
    v.get_label_by_id("110").set_text("217")
    v.get_label_by_id("101").set_text("195")
    v.get_label_by_id("100").set_text("245")
    plt.title(title)
    # plt.annotate('Unknown set', xy=v.get_label_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
    #             ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
    #            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
    plt.show()
    fig.savefig(filename, bbox_inches="tight")
