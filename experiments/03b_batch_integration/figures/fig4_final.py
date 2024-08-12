# figure 3: new data integration
# comparing DGD to MultiVI+scArches on what datasets?

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os
from omicsdgd import DGD
import umap.umap_ as umap
import matplotlib.transforms as mtransforms
from omicsdgd.functions._analysis import discrete_kullback_leibler

#####################
# define model names, directory and batches
#####################
save_dir = "../results/trained_models/"
analysis_dir = "../results/analysis/"
plot_dir = "../results/revision/"
data_name = "human_bonemarrow"
model_names = [
    "human_bonemarrow_l20_h2-3_leftout_site1",
    "human_bonemarrow_l20_h2-3_leftout_site2",
    "human_bonemarrow_l20_h2-3_leftout_site3",
    "human_bonemarrow_l20_h2-3_leftout_site4",
]
mvi_names = [
    "l20_e2_d2_leftout_site1_scarches",
    "l20_e2_d2_leftout_site2_scarches",
    "l20_e2_d2_leftout_site3_scarches",
    "l20_e2_d2_leftout_site4_scarches",
]
batches_left_out = ["site1", "site2", "site3", "site4"]

#####################
#####################
# set up figure
#####################
#####################
figure_height = 8
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
n_cols = 11
n_rows = 4
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=0.6, hspace=8.0)
ax_list = []
# palette_models = ["palegoldenrod", "cornflowerblue"]
palette_models = ["#DAA327", "#015799"]
# palette = ["#BDE1CD", "#40868A"]
# palette = ["#F9EDF5", "#EC6B6B"]
palette = ["#FFD6D6", "#EC6B6B"]
plt.rcParams.update(
    {
        "font.size": 6,
        "axes.linewidth": 0.5,
        "xtick.major.size": 1.5,
        "xtick.major.width": 0.5,
        "ytick.major.size": 1.5,
        "ytick.major.width": 0.5,
    }
)
handletextpad = 0.1
legend_x_dist, legend_y_dist = -0.02, 0.0
grid_letter_positions = [-0.1, 0.1]
grid_letter_fontsize = 8
grid_letter_fontfamily = "sans-serif"
grid_letter_fontweight = "bold"
point_size = 0.5
line_width = 0.5
strip_size = 2
box_width = 0.5
pointplot_scale = 0.5
pointplot_errwidth = 0.7
pointplot_capsize = 0.2
# palette_3colrs = ["lightgray", "darkgrey", "darkmagenta", "darkolivegreen", "firebrick", "midnightblue"]
palette_3colrs = [
    "lightgray",
    "#FFD6D6",
    "darkolivegreen",
    "firebrick",
    "midnightblue",
    "#EC6B6B",
]
# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
# change


#####################
# define functions for calculating error ratios
# and other helper functions
#####################
def return_error_ratios(df_c, model_prefix, site, model_name):
    """returns error ratios for the batches that had been left out of training"""
    df_site = pd.read_csv(
        analysis_dir
        + "batch_integration/"
        + model_prefix
        + "_"
        + site
        + "_prediction_errors.csv"
    )
    df_site = df_site[df_site["batch_id"] == site]
    error_ratios = [
        (
            df_site["error"].values[x]
            / df_c["error"].values[
                np.where(df_c["sample_id"].values == df_site["sample_id"].values[x])[0][
                    0
                ]
            ]
        )
        for x in range(len(df_site))
    ]
    df_error_ratios = pd.DataFrame(
        {
            "error": df_site["error"].values,
            "error_ratio": error_ratios,
            "batch": site,
            "model": model_name,
            "prediction type": "unseen",
        }
    )
    return df_error_ratios


def return_inverse_error_ratios(df_c, model_prefix, site, model_name):
    """returns error ratios for all batches included in training"""
    df_site = pd.read_csv(
        analysis_dir
        + "batch_integration/"
        + model_prefix
        + "_"
        + site
        + "_prediction_errors.csv"
    )
    df_site = df_site[df_site["batch_id"] != site]
    error_ratios = [
        (
            df_site["error"].values[x]
            / df_c["error"].values[
                np.where(df_c["sample_id"].values == df_site["sample_id"].values[x])[0][
                    0
                ]
            ]
        )
        for x in range(len(df_site))
    ]
    df_error_ratios = pd.DataFrame(
        {
            "error": df_site["error"].values,
            "error_ratio": error_ratios,
            "batch": site,
            "model": model_name,
            "prediction type": "seen",
        }
    )
    return df_error_ratios


def get_average_metric(df, df_new, metric):
    """get the mean metric of a different dataframe sorted the way of the first dataframe"""
    out = []
    for i in range(len(df)):
        mod_id = df["model"].values[i]
        if mod_id == "multiVI + scArches":
            mod_id = "multiVI+scArches"
        batch_id = df["batch"].values[i]
        if batch_id == "none":
            mean_val = 1.0
        else:
            mean_val = df_new[
                (df_new["model"] == mod_id) & (df_new["batch"] == batch_id)
            ][metric].mean()
        out.append(mean_val)
    return out


def get_metric(df, df_new, metric):
    """get the mean metric of a different dataframe sorted the way of the first dataframe"""
    out = []
    for i in range(len(df)):
        mod_id = df["model"].values[i]
        if mod_id == "multiVI + scArches":
            mod_id = "multiVI"
        batch_id = df["batch"].values[i]
        if batch_id == "none":
            mean_val = 1.0
        else:
            mean_val = df_new[
                (df_new["model"] == mod_id) & (df_new["batch"] == batch_id)
            ][metric].item()
        out.append(mean_val)
    return out


#####################
# A
# error ratios
# violin plots of the error ratios of each batch
# with color split for whether the predictions came from unseen or seen batches
#####################

###
# preparing the plotting data
###
# calculate for DGD
df_control = pd.read_csv(
    analysis_dir + "batch_integration/human_bonemarrow_none_prediction_errors.csv"
)
for i, site in enumerate(["site1", "site2", "site3", "site4"]):
    # first for unseen batches
    df_error_ratios_temp = return_error_ratios(
        df_control, "human_bonemarrow", site, "multiDGD"
    )
    # add DKL
    df_error_ratios_temp["DKL"] = discrete_kullback_leibler(
        df_control[
            (df_control["batch_id"] == site) & (df_control["model_id"] == "none")
        ]["error"].values
        / 100,  # for binning in rounding process
        df_error_ratios_temp["error"].values / 100,
    )
    if i == 0:
        df_error_ratios = df_error_ratios_temp
    else:
        df_error_ratios = pd.concat([df_error_ratios, df_error_ratios_temp], axis=0)
    # then for seen batches
    df_error_ratios_temp = return_inverse_error_ratios(
        df_control, "human_bonemarrow", site, "multiDGD"
    )
    df_error_ratios_temp["DKL"] = discrete_kullback_leibler(
        df_control[
            (df_control["batch_id"] != site) & (df_control["model_id"] == "none")
        ]["error"].values
        / 100,  # for binning in rounding process
        df_error_ratios_temp["error"].values / 100,
    )
    df_error_ratios = pd.concat([df_error_ratios, df_error_ratios_temp], axis=0)
dkl_dgd = np.asarray(
    [
        discrete_kullback_leibler(
            df_control[df_control["model_id"] == "none"]["error"].values / 100,
            df_error_ratios[(df_error_ratios["batch"] == site)]["error"].values / 100,
        )
        for site in batches_left_out
    ]
)
dkl_dgd = dkl_dgd.mean()
# calculate for mVI+scArches
df_control = pd.read_csv(
    analysis_dir + "batch_integration/mvi_human_bonemarrow_none_prediction_errors.csv"
)
for i, site in enumerate(["site1", "site2", "site3", "site4"]):
    df_error_ratios_temp = return_error_ratios(
        df_control, "mvi_human_bonemarrow", site, "multiVI+scArches"
    )
    df_error_ratios_temp["DKL"] = discrete_kullback_leibler(
        df_control[
            (df_control["batch_id"] == site) & (df_control["model_id"] == "none")
        ]["error"].values
        / 100,  # for binning in rounding process
        df_error_ratios_temp["error"].values / 100,
    )
    df_error_ratios = pd.concat([df_error_ratios, df_error_ratios_temp], axis=0)
    df_error_ratios_temp = return_inverse_error_ratios(
        df_control, "mvi_human_bonemarrow", site, "multiVI+scArches"
    )
    df_error_ratios_temp["DKL"] = discrete_kullback_leibler(
        df_control[
            (df_control["batch_id"] != site) & (df_control["model_id"] == "none")
        ]["error"].values
        / 100,  # for binning in rounding process
        df_error_ratios_temp["error"].values / 100,
    )
    df_error_ratios = pd.concat([df_error_ratios, df_error_ratios_temp], axis=0)
dkl_mvi = np.asarray(
    [
        discrete_kullback_leibler(
            df_control[df_control["model_id"] == "none"]["error"].values / 100,
            df_error_ratios[(df_error_ratios["batch"] == site)]["error"].values / 100,
        )
        for site in batches_left_out
    ]
)
dkl_mvi = dkl_mvi.mean()
df_error_ratios["prediction type"] = df_error_ratios["prediction type"].astype(
    "category"
)
df_error_ratios["prediction type"].cat.set_categories(["seen", "unseen"], inplace=True)

for i, site in enumerate(["site1", "site2", "site3", "site4"]):
    df_error_ratios_temp = df_error_ratios[(df_error_ratios["model"] == "multiDGD") & (df_error_ratios["batch"] == site)]
    seen_len = len(df_error_ratios_temp[df_error_ratios_temp["prediction type"] == "seen"])
    unseen_len = len(df_error_ratios_temp[df_error_ratios_temp["prediction type"] == "unseen"])
    print(f"{site} seen: {seen_len}, unseen: {unseen_len}")

###
# violin plot
###
ax_list.append(plt.subplot(gs[0:2, 0:7]))
ax_list[-1].text(
    grid_letter_positions[0] + 0.025,
    1.0 + grid_letter_positions[1],
    "A",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)

"""
ax_list[-1].text(
    grid_letter_positions[0] + 0.745,
    1.3 + grid_letter_positions[1],
    "Integrating novel batches",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
"""

ax_list[-1].axhline(1.0, color="gray", linestyle="--", linewidth=line_width * 2)
sns.violinplot(
    data=df_error_ratios[df_error_ratios["model"] == "multiDGD"],
    x="batch",
    y="error_ratio",
    hue="prediction type",
    split=True,
    inner="quart",
    linewidth=line_width,
    linecolor="gray",
    palette=palette,
    ax=ax_list[-1],
)

ax_list[-1].set_ylabel("Test loss ratio")
ax_list[-1].set_xlabel("Batch")
ax_list[-1].set_title("Relative test prediction by batch")
ax_list[-1].legend(
    bbox_to_anchor=(0.62, 0.21),
    loc="upper left",
    frameon=False,
    handletextpad=handletextpad * 4,
    ncol=2,
)  # .set_visible(False)
# sort DKLs for annotation
dkl_annot = []
for i, site in enumerate(batches_left_out):
    dkl_annot.append(
        df_error_ratios[
            (df_error_ratios["batch"] == site)
            & (df_error_ratios["model"] == "multiDGD")
            & (df_error_ratios["prediction type"] == "seen")
        ]["DKL"].values[0]
    )
    dkl_annot.append(
        df_error_ratios[
            (df_error_ratios["batch"] == site)
            & (df_error_ratios["model"] == "multiDGD")
            & (df_error_ratios["prediction type"] == "unseen")
        ]["DKL"].values[0]
    )
for i in range(len(dkl_annot)):
    if i % 2 == 0:
        ax_list[-1].text(
            int(i / 2) - 0.35, 1.035, "(" + str(round(dkl_annot[i], 2)) + ", "
        )
    else:
        ax_list[-1].text(int(i / 2) + 0.05, 1.035, str(round(dkl_annot[i], 2)) + ")")

#####################
# B
# comparisons between DGD and mVI+scArches
# even though they are not comparable
#####################

###
# prepare the plotting data
###

# get all the performance metrics and batch effect metrics
metrics_df = pd.read_csv(
    analysis_dir + "batch_integration/human_bonemarrow_reconstruction_performance.csv"
)
metrics_df2 = pd.read_csv(
    analysis_dir + "batch_integration/human_bonemarrow_reconstruction_performance_revision.csv"
)
df_batch_effect = pd.read_csv(
    analysis_dir + "batch_integration/human_bonemarrow_batch_effect.csv"
)
df_batch_effect["1 - ASW"] = 1 - df_batch_effect["ASW"]
df_batch_effect["(1 - ASW) ratio"] = (
    df_batch_effect["1 - ASW"] / df_batch_effect["1 - ASW"].values[0]
)
df_batch_effect["AUPRC"] = 0.279039
df_batch_effect["AUPRC"][1:] = metrics_df2["AUPRC"].values
df_batch_effect_2 = pd.read_csv(
    analysis_dir + "batch_integration/human_bonemarrow_batch_effect_mvi.csv"
)
df_batch_effect_2["1 - ASW"] = 1 - df_batch_effect_2["ASW"]
df_batch_effect_2["(1 - ASW) ratio"] = (
    df_batch_effect_2["1 - ASW"] / df_batch_effect_2["1 - ASW"].values[0]
)
#print(df_batch_effect_2)
# these values are taken from the mvi_performance notebook and are in order with the models
#0.258541
df_batch_effect_2["AUPRC"] = [0.26081529699812717, 0.25975587259993305, 0.25924887119849777, 0.25766518018835755, 0.25905025997182063]
df_batch_effect = pd.concat([df_batch_effect, df_batch_effect_2], axis=0)
df_batch_effect["test error ratio"] = get_average_metric(
    df_batch_effect, df_error_ratios, "error_ratio"
)
df_batch_effect["RMSE (rna)"] = get_metric(df_batch_effect, metrics_df, "RMSE (rna)")
df_batch_effect["balanced accuracy"] = get_metric(
    df_batch_effect, metrics_df, "balanced accuracy"
)
# ensure typical order of models by categorical
df_error_ratios["model"] = [
    x if x == "multiDGD" else "MultiVI\n+scArches"
    for x in df_error_ratios["model"].values
]
df_error_ratios["model"] = df_error_ratios["model"].astype("category")
df_error_ratios["model"].cat.set_categories(
    ["MultiVI\n+scArches", "multiDGD"], inplace=True
)
df_batch_effect["model"] = [
    x if x == "multiDGD" else "MultiVI\n+scArches"
    for x in df_batch_effect["model"].values
]
df_batch_effect["model"] = df_batch_effect["model"].astype("category")
df_batch_effect["model"].cat.set_categories(
    ["MultiVI\n+scArches", "multiDGD"], inplace=True
)

###
# plots
###

# test error ratio
ax_list.append(plt.subplot(gs[0:2, 8:11]))
ax_list[-1].text(
    grid_letter_positions[0] - 0.1,
    1.0 + grid_letter_positions[1],
    "B",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
ax_list[-1].axhline(1.0, color="gray", linestyle="--", linewidth=line_width * 2)
metric = "error_ratio"
# sns.boxplot(data=df_error_ratios,#[df_error_ratios["prediction type"] == "unseen"],
#            x="model", y=metric, hue="model", ax=ax_list[-1], palette=palette, dodge=False)
sns.violinplot(
    data=df_error_ratios,  # [df_error_ratios["prediction type"] == "unseen"],
    x="model",
    y=metric,
    hue="model",
    ax=ax_list[-1],
    linewidth=line_width,
    linecolor="gray",
    palette=palette_models,
    dodge=False,
)
#print(df_error_ratios[df_error_ratios["model"] == "multiDGD"])
test_len = len(df_error_ratios[df_error_ratios["model"] == "multiDGD"])
print(f"{test_len} samples in plot b")
print(test_len/4)
ax_list[-1].legend(
    bbox_to_anchor=(1.02 + legend_x_dist, 1.0 + legend_y_dist),
    loc="upper left",
    frameon=False,
    handletextpad=handletextpad,
    ncol=1,
).set_visible(False)
ax_list[-1].set_ylabel("Test loss ratio")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("Relative reconstruction\nerror")
# add overall DKLs
ax_list[-1].set_ylim(0.85, 1.18)
ax_list[-1].text(0.0, 1.14, round(dkl_mvi, 2),horizontalalignment='center',verticalalignment='center')
ax_list[-1].text(1.0, 1.14, round(dkl_dgd, 2),horizontalalignment='center',verticalalignment='center')

# RMSE
ax_list.append(plt.subplot(gs[2:4, :3]))
ax_list[-1].text(
    grid_letter_positions[0] - 0.1,
    1.1 + grid_letter_positions[1],
    "C",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
metric = "RMSE (rna)"
ax_list[-1].axhline(
    1.068006157875061, color=palette_models[0],
    linewidth=line_width * 1.5,
    linestyle="--",
    alpha=0.7,
)
ax_list[-1].axhline(
    0.8810019493103027,
    color=palette_models[1],
    linewidth=line_width * 1.5,
    linestyle="--",
    alpha=0.7,
)
sns.pointplot(
    data=df_batch_effect[df_batch_effect["batch"] != "none"],
    x="model",
    y=metric,
    hue="model",
    ax=ax_list[-1],
    palette=palette_models,
    errorbar="se",
    dodge=False,
    markers=".",
    linestyles="",
    scale=pointplot_scale,
    errwidth=pointplot_errwidth,
    capsize=pointplot_capsize,
)
sns.stripplot(
    x="model",
    y=metric,
    color="black",
    data=df_batch_effect[df_batch_effect["batch"] != "none"],
    ax=ax_list[-1],
    size=strip_size,
)
reconstruction_temp = pd.read_csv(
    analysis_dir + "performance_evaluation/reconstruction/human_bonemarrow.csv"
)
ax_list[-1].legend(
    bbox_to_anchor=(1.02 + legend_x_dist, 1.0 + legend_y_dist),
    loc="upper left",
    frameon=False,
    handletextpad=handletextpad,
    ncol=1,
).set_visible(False)
ax_list[-1].set_ylabel("RMSE")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("Reconstruction\nperformance (RNA) \u2193")

# balanced accuracy
ax_list.append(plt.subplot(gs[2:4, 4:7]))
#metric = "balanced accuracy"
metric = "AUPRC"
ax_list[-1].axhline(
    0.258541, color=palette_models[0],
    linewidth=line_width * 1.5,
    linestyle="--",
    alpha=0.7,
)
ax_list[-1].axhline(
    0.279039, color=palette_models[1],
    linewidth=line_width * 1.5,
    linestyle="--",
    alpha=0.7,
)
sns.pointplot(
    data=df_batch_effect[df_batch_effect["batch"] != "none"],
    x="model",
    y=metric,
    hue="model",
    ax=ax_list[-1],
    palette=palette_models,
    errorbar="se",
    dodge=False,
    markers=".",
    linestyles="",
    scale=pointplot_scale,
    errwidth=pointplot_errwidth,
    capsize=pointplot_capsize,
)
sns.stripplot(
    x="model",
    y=metric,
    color="black",
    data=df_batch_effect[df_batch_effect["batch"] != "none"],
    ax=ax_list[-1],
    size=strip_size,
)
ax_list[-1].legend(
    bbox_to_anchor=(1.02 + legend_x_dist, 1.0 + legend_y_dist),
    loc="upper left",
    frameon=False,
    handletextpad=handletextpad * 4,
    ncol=1,
).set_visible(False)
ax_list[-1].set_ylabel("AUPRC")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("Reconstruction\nperformance (ATAC) \u2191")

# batch effect removal
ax_list.append(plt.subplot(gs[2:4, 8:11]))
metric = "1 - ASW"
ax_list[-1].axhline(
    df_batch_effect[
        (df_batch_effect["model"] != "multiDGD") & (df_batch_effect["batch"] == "none")
    ][metric].mean(),
    color=palette_models[0],
    linewidth=line_width * 1.5,
    linestyle="--",
    alpha=0.7,
)
ax_list[-1].axhline(
    df_batch_effect[
        (df_batch_effect["model"] == "multiDGD") & (df_batch_effect["batch"] == "none")
    ][metric].mean(),
    color=palette_models[1],
    linewidth=line_width * 1.5,
    linestyle="--",
    alpha=0.7,
)
sns.pointplot(
    data=df_batch_effect[df_batch_effect["batch"] != "none"],
    x="model",
    y=metric,
    hue="model",
    ax=ax_list[-1],
    palette=palette_models,
    errorbar="se",
    dodge=False,
    markers=".",
    linestyles="",
    scale=pointplot_scale,
    errwidth=pointplot_errwidth,
    capsize=pointplot_capsize,
)
sns.stripplot(
    x="model",
    y=metric,
    color="black",
    data=df_batch_effect[df_batch_effect["batch"] != "none"],
    ax=ax_list[-1],
    size=strip_size,
)
ax_list[-1].legend(
    bbox_to_anchor=(1.02 + legend_x_dist, 1.0 + legend_y_dist),
    loc="upper left",
    frameon=False,
    handletextpad=handletextpad,
    ncol=1,
).set_visible(False)
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("Batch effect removal \u2191")



plt.savefig(
    "../results/figures/fig4_final.pdf",
    dpi=300,
    bbox_inches="tight",
    format="pdf",
)