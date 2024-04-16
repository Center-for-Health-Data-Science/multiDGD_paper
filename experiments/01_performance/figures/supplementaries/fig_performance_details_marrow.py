import os
import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as PathEffects

# set up figure
figure_height = 18
n_cols = 4
n_rows = 10
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=0.7, hspace=200.0)
ax_list = []
palette_2colrs = ["#015799", "#DAA327"]
plt.rcParams.update(
    {
        "font.size": 6,
        "axes.linewidth": 0.3,
        "xtick.major.size": 1.5,
        "xtick.major.width": 0.3,
        "ytick.major.size": 1.5,
        "ytick.major.width": 0.3,
        #"lines.linewidth": 0.5,
    }
)
handletextpad = 0.1
legend_x_dist, legend_y_dist = -0.0, 0.0
grid_letter_positions = [-0.1, 0.05]
grid_letter_fontsize = 8
grid_letter_fontfamily = "sans-serif"
grid_letter_fontweight = "bold"
heatmap_fontsize = 4
point_size = 0.2
linewidth = 0.5
alpha = 1
point_linewidth = 0.0
handlesize = 0.3

# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)

####################
# plot sample-wise
####################

# get data

random_seeds = [0, 37, 8790]
model_types = ["", "", ""]
n_components = [22, 22, 22]
model_descriptors = ["default", "default", "default"]
n_features_bm = 129921
modality_switch = 13431
data_name = "human_bonemarrow"
result_path = "../results/revision/analysis/batch_integration/" + data_name + "/"
plot_path = "../results/revision/plots/"

if not os.path.exists(result_path + "celltypes_test.csv"):
    import anndata as ad
    data = ad.read_h5ad("../../data/" + data_name + ".h5ad")
    data.X = data.layers["counts"]
    test_indices = list(np.where(data.obs["train_val_test"] == "test")[0])
    celltypes = data.obs["cell_type"].values[test_indices]
    # save the celltypes as csv
    pd.DataFrame(celltypes).to_csv(result_path + "celltypes_test.csv", index=False, header=False)
    mean_expression = np.asarray(data.X.mean(axis=0)).flatten()
    np.save(result_path + "mean_expression_test.npy", mean_expression)
    data = None
else:
    celltypes = pd.read_csv(result_path + "celltypes_test.csv", header=None).values.flatten()
    mean_expression = np.load(result_path + "mean_expression_test.npy")

# load all prediction errors
prediction_errors_sample = pd.DataFrame()
for i, model_name in enumerate(model_descriptors):
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_"
        + model_types[i]
        + "_ncomp"
        + str(n_components[i])
        + "_errors_samplewise.csv"
    )
    temp_df["rs"] = random_seeds[i]
    temp_df["celltype"] = celltypes
    #temp_df["count depth"] = mean_expression_cell
    prediction_errors_sample = pd.concat([prediction_errors_sample, temp_df])
prediction_errors_sample["error"] = prediction_errors_sample["rna_mean"].values + prediction_errors_sample["atac_mean"].values

# load all prediction errors
prediction_metrics_sample = pd.DataFrame()
for i, model_name in enumerate(model_descriptors):
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_"
        + model_types[i]
        + "_ncomp"
        + str(n_components[i])
        + "_RMSE-BA_samplewise.csv"
    )
    # only keep columns rmse and batch
    temp_df["rs"] = random_seeds[i]
    temp_df["celltype"] = celltypes
    #temp_df["count depth"] = mean_expression_cell
    temp_df["Model"] = "multiDGD"
    prediction_metrics_sample = pd.concat([prediction_metrics_sample, temp_df])
    # also load mvi
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_RMSE-BA_samplewise_mvi.csv"
    )
    temp_df["rs"] = random_seeds[i]
    temp_df["celltype"] = celltypes
    #temp_df["count depth"] = mean_expression_cell
    temp_df["Model"] = "MultiVI"
    prediction_metrics_sample = pd.concat([prediction_metrics_sample, temp_df])

# make grouped statistics (mean and standard error) for ba, auprc, spearman

n_samples = prediction_metrics_sample.groupby(["Model"]).size().values[0]

summary_stats_sample = prediction_metrics_sample.groupby(["Model"]).agg(
    mean_ba=("ba", "mean"),
    std_ba=("ba", "std"),
    mean_rmse=("rmse", "mean"),
    std_rmse=("rmse", "std"),
    mean_auprc=("auprc", "mean"),
    std_auprc=("auprc", "std"),
    mean_spearman=("spearman", "mean"),
    std_spearman=("spearman", "std"),
)

# make new columns for sem
summary_stats_sample["sem_ba"] = summary_stats_sample["std_ba"] / np.sqrt(n_samples)
summary_stats_sample["sem_rmse"] = summary_stats_sample["std_rmse"] / np.sqrt(n_samples)
summary_stats_sample["sem_auprc"] = summary_stats_sample["std_auprc"] / np.sqrt(n_samples)
summary_stats_sample["sem_spearman"] = summary_stats_sample["std_spearman"] / np.sqrt(n_samples)

####################

# a figure with 3 subplots, showing the three different metrics for the ATAC test performance

# RMSE
ax_list.append(fig.add_subplot(gs[0:3, 0]))
"""
ax_list[-1].text(
    2.55,
    1.3,
    "Reconstruction performance",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
"""
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.0 + 2 * grid_letter_positions[1],
    "A",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
sns.violinplot(
    x="Model",
    y="rmse",
    data=prediction_metrics_sample,
    ax=ax_list[-1],
    scale="width",
    inner="quartile",
    linewidth=0.5,
    palette=palette_2colrs,
)
# print the mean and standard error above the violin plots (go by the order of the models in the plot)
for i, model in enumerate(ax_list[-1].get_xticklabels()):
    model = model.get_text()
    ax_list[-1].text(i, 8.5, f"{summary_stats_sample[summary_stats_sample.index == model]['mean_rmse'].item():.2f} ± {summary_stats_sample[summary_stats_sample.index == model]['sem_rmse'].item():.2f}", ha="center", va="bottom", fontsize=5)
#ax_list[-1].set_yscale("log")
ax_list[-1].set_ylim(0, 10)
ax_list[-1].set_ylabel("RMSE (sample-wise)")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("RNA (per cell) \u2193")

# next auprc
ax_list.append(fig.add_subplot(gs[0:3, 1]))
fraction_pos = 0.030785650582849974
ax_list[-1].axhline(y=fraction_pos, color="darkmagenta", linestyle="--", linewidth=0.5)
sns.violinplot(
    x="Model",
    y="auprc",
    data=prediction_metrics_sample,
    ax=ax_list[-1],
    scale="width",
    inner="quartile",
    linewidth=0.5,
    palette=palette_2colrs,
)
ax_list[-1].set_ylim(0, 1)
for i, model in enumerate(ax_list[-1].get_xticklabels()):
    model = model.get_text()
    ax_list[-1].text(i, 0.85, f"{summary_stats_sample[summary_stats_sample.index == model]['mean_auprc'].item():.2f} ± {summary_stats_sample[summary_stats_sample.index == model]['sem_auprc'].item():.2f}", ha="center", va="bottom", fontsize=5)
ax_list[-1].set_ylabel("AUPRC (sample-wise)")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("ATAC (per cell) \u2191")

# in the second row, we plot the RMSE and BA as violinplots over cell types
# first the RMSE
ax_list.append(fig.add_subplot(gs[3:6, 0:2]))
ax_list[-1].text(
    grid_letter_positions[0] + 0.025,
    1.0 + 2 * grid_letter_positions[1],
    "C",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
# make these violin plots half for the different models
sns.violinplot(
    x="celltype",
    y="rmse",
    data=prediction_metrics_sample,
    hue="Model",
    split=True,
    ax=ax_list[-1],
    scale="width",
    inner="quartile",
    linewidth=0.1,
    palette=palette_2colrs,
)
ax_list[-1].set_ylim(0, 10)
#ax_list[-1].set_yscale("log")
ax_list[-1].set_ylabel("RMSE (sample-wise)")
ax_list[-1].set_xlabel("Cell type")
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=90)
ax_list[-1].set_title("RNA (per cell, by cell type) \u2193")
# remove legend
ax_list[-1].get_legend().remove()
# now the BA
ax_list.append(fig.add_subplot(gs[3:6, 2:]))
sns.violinplot(
    x="celltype",
    y="auprc",
    data=prediction_metrics_sample,
    hue="Model",
    split=True,
    ax=ax_list[-1],
    scale="width",
    inner="quartile",
    linewidth=0.1,
    palette=palette_2colrs,
)
ax_list[-1].set_ylim(0, 1)
ax_list[-1].set_ylabel("AUPRC (sample-wise)")
ax_list[-1].set_xlabel("Cell type")
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=90)
ax_list[-1].set_title("ATAC (per cell, by cell type) \u2191")
# set legend in the right lower corner
ax_list[-1].legend(
    title="Model",
    bbox_to_anchor=(1.02, 1),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad * 2,
).remove()

####################
# plot feature-wise
####################

# get data

# !!! I don't know how this looks here, need to check first

# load all prediction errors
prediction_errors_gene = pd.DataFrame()
for i, model_name in enumerate(model_descriptors):
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_"
        + model_types[i]
        + "_ncomp"
        + str(n_components[i])
        + "_errors_genewise.csv"
    )
    temp_df["rs"] = random_seeds[i]
    #temp_df["mean"] = mean_expression
    prediction_errors_gene = pd.concat([prediction_errors_gene, temp_df])
#prediction_errors["normalized error"] = prediction_errors["error"] / n_features_bm
prediction_errors_gene["error"] = [prediction_errors_gene["rna_mean"].values[i] if prediction_errors_gene["modality"].values[i] == "rna" else prediction_errors_gene["atac_mean"].values[i] for i in range(prediction_errors_gene.shape[0])]

# !!! I don't know how this looks here, need to check first

# load all prediction errors
prediction_metrics_gene = pd.DataFrame()
for i, model_name in enumerate(model_descriptors):
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_"
        + model_types[i]
        + "_ncomp"
        + str(n_components[i])
        + "_RMSE-BA_genewise.csv"
    )
    temp_df["rs"] = random_seeds[i]
    temp_df["mean"] = mean_expression
    temp_df["Model"] = "multiDGD"
    prediction_metrics_gene = pd.concat([prediction_metrics_gene, temp_df])
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_RMSE-BA_genewise_mvi.csv"
    )
    temp_df["rs"] = random_seeds[i]
    temp_df["mean"] = mean_expression
    temp_df["Model"] = "MultiVI"
    prediction_metrics_gene = pd.concat([prediction_metrics_gene, temp_df])
#prediction_errors["normalized error"] = prediction_errors["error"] / n_features_bm

# make grouped statistics (mean and standard error) for ba, auprc, spearman

n_features = prediction_metrics_gene.groupby(["Model"]).size().values[0]

summary_stats_gene = prediction_metrics_gene.groupby(["Model"]).agg(
    mean_ba=("ba", "mean"),
    std_ba=("ba", "std"),
    mean_rmse=("rmse", "mean"),
    std_rmse=("rmse", "std"),
    mean_auprc=("auprc", "mean"),
    std_auprc=("auprc", "std"),
    mean_spearman=("spearman", "mean"),
    std_spearman=("spearman", "std"),
)

# make new columns for sem
summary_stats_gene["sem_ba"] = summary_stats_gene["std_ba"] / np.sqrt(n_features)
summary_stats_gene["sem_rmse"] = summary_stats_gene["std_rmse"] / np.sqrt(n_features)
summary_stats_gene["sem_auprc"] = summary_stats_gene["std_auprc"] / np.sqrt(n_features)
summary_stats_gene["sem_spearman"] = summary_stats_gene["std_spearman"] / np.sqrt(n_features)

rna_bins, rna_bin_ranges = pd.qcut(prediction_metrics_gene[prediction_metrics_gene["feature"]=="rna"]["mean"], q=10, labels=False, retbins=True)
atac_bins, atac_bin_ranges = pd.qcut(prediction_metrics_gene[prediction_metrics_gene["feature"]=="atac"]["mean"], q=10, labels=False, retbins=True)
prediction_metrics_gene["mean_bin"] = 0
prediction_metrics_gene.loc[prediction_metrics_gene["feature"]=="rna", "mean_bin"] = rna_bins
prediction_metrics_gene.loc[prediction_metrics_gene["feature"]=="atac", "mean_bin"] = atac_bins

mean_rna = prediction_metrics_gene[prediction_metrics_gene["feature"]=="rna"]["mean"].mean()
mean_atac = prediction_metrics_gene[prediction_metrics_gene["feature"]=="atac"]["mean"].mean()

####################

# RMSE
ax_list.append(fig.add_subplot(gs[0:3, 2]))
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.0 + 2 * grid_letter_positions[1],
    "B",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
#"""
sns.violinplot(
    x="Model",
    y="rmse",
    data=prediction_metrics_gene,
    ax=ax_list[-1],
    scale="width",
    inner="quartile",
    linewidth=0.5,
    palette=palette_2colrs,
)
"""
sns.boxplot(
    x="Model",
    y="rmse",
    data=prediction_metrics_gene,
    ax=ax_list[-1],
    linewidth=0.5,
    palette=palette_2colrs,
)
"""
# print the mean and standard error above the violin plots (go by the order of the models in the plot)
for i, model in enumerate(ax_list[-1].get_xticklabels()):
    model = model.get_text()
    ax_list[-1].text(i, 85*2, f"{summary_stats_gene[summary_stats_gene.index == model]['mean_rmse'].item():.2f} ± {summary_stats_gene[summary_stats_gene.index == model]['sem_rmse'].item():.2f}", ha="center", va="bottom", fontsize=5)
ax_list[-1].set_ylim(0, 200)
ax_list[-1].set_ylabel("RMSE (gene-wise)")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("RNA (per gene) \u2193")

# next auprc
ax_list.append(fig.add_subplot(gs[0:3, 3]))
ax_list[-1].axhline(y=fraction_pos, color="darkmagenta", linestyle="--", linewidth=0.5)
sns.violinplot(
    x="Model",
    y="auprc",
    data=prediction_metrics_gene,
    ax=ax_list[-1],
    scale="width",
    inner="quartile",
    linewidth=0.5,
    palette=palette_2colrs,
)
ax_list[-1].set_ylim(0, 1)
for i, model in enumerate(ax_list[-1].get_xticklabels()):
    model = model.get_text()
    ax_list[-1].text(i, 0.85, f"{summary_stats_gene[summary_stats_gene.index == model]['mean_auprc'].item():.2f} ± {summary_stats_gene[summary_stats_gene.index == model]['sem_auprc'].item():.2f}", ha="center", va="bottom", fontsize=5)
ax_list[-1].set_ylabel("AUPRC (peak-wise)")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("ATAC (per peak) \u2191")

# below add a row where each of the metrics is plotted against the mean expression of the features
# first the BA
ax_list.append(fig.add_subplot(gs[7:, 0:2]))
ax_list[-1].text(
    grid_letter_positions[0] + 0.025,
    1.0 + 2 * grid_letter_positions[1],
    "D",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
# plot the RMSE as violinplots over the mean expression of the features (as bins)
sns.violinplot(
    x="mean_bin",
    y="rmse",
    data=prediction_metrics_gene,
    hue="Model",
    split=True,
    ax=ax_list[-1],
    scale="width",
    inner="quartile",
    linewidth=0.1,
    palette=palette_2colrs,
)
# set xticklabels to rna_bin_ranges
ax_list[-1].set_xticklabels([f"{rna_bin_ranges[i]:.3f}-{rna_bin_ranges[i+1]:.3f}" for i in range(len(rna_bin_ranges)-1)], rotation=90)
ax_list[-1].set_yscale("log")
ax_list[-1].set_ylim(0, 100)
ax_list[-1].set_ylabel("RMSE (gene-wise)")
ax_list[-1].set_xlabel("Average count")
ax_list[-1].set_title("RNA (per gene, binned) \u2193")
ax_list[-1].get_legend().remove()

# next the AUPRC
ax_list.append(fig.add_subplot(gs[7:, 2:]))
ax_list[-1].axhline(y=fraction_pos, color="darkmagenta", linestyle="--", linewidth=0.5)
sns.violinplot(
    x="mean_bin",
    y="auprc",
    data=prediction_metrics_gene,
    hue="Model",
    split=True,
    ax=ax_list[-1],
    scale="width",
    inner="quartile",
    linewidth=0.1,
    palette=palette_2colrs,
)
ax_list[-1].set_xticklabels([f"{atac_bin_ranges[i]:.3f}-{atac_bin_ranges[i+1]:.3f}" for i in range(len(atac_bin_ranges)-1)], rotation=90)
ax_list[-1].set_ylim(0, 1)
ax_list[-1].set_ylabel("AUPRC (peak-wise)")
ax_list[-1].set_xlabel("Average count")
ax_list[-1].set_title("ATAC (per peak, binned) \u2191")
ax_list[-1].legend(
    title="model",
    bbox_to_anchor=(-0.4, -0.6),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad * 2,
    ncol=2,
)

# save
fig.savefig(os.path.join(plot_path, "performance_human_bonemarrow_metrics_sample-and-feature-wise.png"), bbox_inches="tight", dpi=300)