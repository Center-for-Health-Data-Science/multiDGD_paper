# imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
#from sklearn.metrics import silhouette_score

result_dir = "../results/revision/analysis/"
plot_dir = "../results/revision/plots/"

cm = 1 / 2.54

import matplotlib.gridspec as gridspec

batch_palette = ["#EEE7A8", "cornflowerblue", "darkmagenta", "darkslategray"]
stage_palette = "magma_r"
palette_3colrs = ["#DAA327", "#BDE1CD", "#015799"]

plt.rcParams.update(
    {
        "font.size": 6,
        "axes.linewidth": 0.3,
        "xtick.major.size": 1.5,
        "xtick.major.width": 0.3,
        "ytick.major.size": 1.5,
        "ytick.major.width": 0.3,
    }
)

handletextpad = 0.1
point_size = 0.5
linewidth = 0.2
alpha = 0.1
point_linewidth = 0.0
handlesize = 0.3
dodge = True

# for every subset, load the mvi clustering dataframes

subset_samples = [567, 5671, 14178, 28357, 42535, 56714]
for subset in subset_samples:
    df = pd.read_csv(
        "../results/revision/analysis/performance/data_efficiency_clustering_mvi_subset"
        + str(subset)
        + ".csv"
    )
    df = df.drop(columns=["Unnamed: 0"])
    df["Number of cells"] = df["n_samples"]
    df = df.drop(columns=["n_samples"])
    df["Subset size"] = subset
    if subset == 567:
        df_mvi = df
    else:
        if subset == 42535:
            # drop the accidental subset 56714
            df = df[df["Number of cells"] != 56714]
        df_mvi = pd.concat([df_mvi, df])

# drop everything except ARI and Number of cells
df_mvi = df_mvi[["ARI", "Number of cells"]]
df_mvi["Model"] = "MultiVI"

# read in the csv file "wandb_subset_ARIs.csv"

df = pd.read_csv(result_dir + "performance/wandb_subset_ARIs.csv", sep=",")
# remove rows 8-10
df = df.drop([8, 9, 10])

# make a column for the number of cells (these are given in the last part of the run name after "subset")
df["last_name"] = [x.split("_")[-1] for x in df["Name"]]
# if subset is not the first part of the last name, then the number of cells is 56714
df["n_cells"] = [x[6:] if "subset" in x else 56714 for x in df["last_name"]]
df["n_cells"] = df["n_cells"].astype(int)

# remove columns Name and Index
df = df.drop(columns=["Name", "Index", "_wandb", "last_name"])
# rename columns
df = df.rename(columns={"AdjustedRandIndex_fromMetaLabel": "ARI", "n_cells": "Number of cells"})

df["Model"] = "multiDGD"

# combine the dataframes

df2 = pd.concat([df_mvi, df])
df2["fraction"] = ((df2["Number of cells"] / 56714 * 100) + 1).astype(int)
# change 101 to 100
df2["fraction"] = df2["fraction"].replace(101, 100)

# plot the ARI over number of cells

pointplot_scale = 0.5
pointplot_errwidth = 0.7
pointplot_capsize = 0.2
palette_2colrs = ["#DAA327", "#015799"]

fig, ax = plt.subplots(figsize=(8 * cm, 6 * cm))
sns.pointplot(
    x="fraction",
    y="ARI",
    data=df2,
    ax=ax,
    hue="Model",
    dodge=dodge,
    linestyles="",
    scale=pointplot_scale,
    errwidth=pointplot_errwidth,
    capsize=pointplot_capsize,
    palette=palette_2colrs,
    errorbar="se"
)

def percent_to_n(x):
    print(x)
    print(type(x))
    return int(x * 0.01 * 56714)
def n_to_percent(x):
    return int(x / 56714 * 100)
#secax = ax_list[-1].secondary_xaxis('top', functions=(percent_to_n, n_to_percent))
#secax.set_xlabel('Number of training samples')
ax2 = ax.twiny()
sns.pointplot(
    x="Number of cells",
    y="ARI",
    data=df2,
    ax=ax2,
    hue="Model",
    linestyles="",
    dodge=dodge,
    scale=pointplot_scale,
    errwidth=pointplot_errwidth,
    capsize=pointplot_capsize,
    palette=palette_2colrs,
    errorbar="se"
)
sns.stripplot(
    x="Number of cells",
    y="ARI",
    data=df2,
    ax=ax2,
    hue="Model",
    color="black",
    dodge=dodge,
    size=1.5
)
ax2.legend_.remove()
ax.legend(bbox_to_anchor=(1.0, 1.05),
                   loc='upper left', frameon=False,title='model',
                   markerscale=handlesize*3,
                   handletextpad=handletextpad*2)#.set_visible(False)
#ax_list[-1].text(30, 1.01, 'placeholder', fontdict={'color': 'red'})
ax.set_xlabel('Percentage of training set')
ax2.set_xlabel('Number of training samples')
ax.set_ylim(0, 1)

# save this figure
fig.savefig(plot_dir + "data_efficiency_clustering_mvi_multiDGD.png", dpi=300, bbox_inches="tight")