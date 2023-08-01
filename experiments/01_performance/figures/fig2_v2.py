"""
Ver extensive figure for results regarding performance evaluation
"""

import os
import pandas as pd
from omicsdgd import DGD
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as PathEffects
from omicsdgd.functions._analysis import (
    make_palette_from_meta,
    gmm_make_confusion_matrix,
)

save_dir = "results/trained_models/"
data_name = "human_bonemarrow"
model_name = "human_bonemarrow_l20_h2-3_test10e"  # this is just a speficiation of the model after test set inference

# set up figure
figure_height = 18
n_cols = 8
n_rows = 12
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=6.0, hspace=40.0)
ax_list = []
palette_2colrs = ["#DAA327", "#015799"]
palette_models = ["#DAA327", "palegoldenrod", "#BDE1CD", "#015799"]
palette_3colrs = ["#DAA327", "#BDE1CD", "#015799"]
batch_palette = ["#EEE7A8", "cornflowerblue", "darkmagenta", "darkslategray"]
extra_palette = ["gray", "darkslategray", "#EEE7A8", "#BDE1CD"]
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

####################################
####################################
# row 1: performance metrics (comparison to multiVI)
####################################
####################################
# load needed analysis data
clustering_df = pd.read_csv(
    "results/analysis/performance_evaluation/clustering_and_batch_effects_multiome.csv"
)
clustering_df["ARI"] = clustering_df["ARI"].round(2)
clustering_df["ASW"] = clustering_df["ASW"].round(2)
clustering_cobolt = pd.read_csv(
    "results/analysis/performance_evaluation/cobolt_human_brain_aris.csv"
)
clustering_cobolt["data"] = "brain (H)"
clustering_cobolt["ARI"] = clustering_cobolt["ARI"].round(2)
multimodel_clustering = pd.concat([clustering_df, clustering_cobolt], axis=0)
clustering_scmm = pd.read_csv(
    "results/analysis/performance_evaluation/scmm_human_brain_aris.csv"
)
clustering_scmm["ARI"] = clustering_scmm["ARI"].round(2)
multimodel_clustering = pd.concat([multimodel_clustering, clustering_scmm], axis=0)
multimodel_clustering["data"] = [
    x.split(" (")[0] for x in multimodel_clustering["data"].values
]
multimodel_clustering["data"] = [
    "marrow" if x == "bone marrow" else x for x in multimodel_clustering["data"].values
]

# change multiVI to MultiVI and cobolt to Cobolt
multimodel_clustering["model"] = [
    "MultiVI" if x == "multiVI" else x for x in multimodel_clustering["model"].values
]
multimodel_clustering["model"] = [
    "Cobolt" if x == "cobolt" else x for x in multimodel_clustering["model"].values
]
multimodel_clustering["model"] = multimodel_clustering["model"].astype("category")
multimodel_clustering["model"] = multimodel_clustering["model"].cat.set_categories(
    ["MultiVI", "Cobolt", "scMM", "multiDGD"]
)

reconstruction_temp = pd.read_csv(
    "results/analysis/performance_evaluation/reconstruction/human_bonemarrow.csv"
)
reconstruction_df = reconstruction_temp
reconstruction_temp["data"] = "marrow"
reconstruction_temp = pd.read_csv(
    "results/analysis/performance_evaluation/reconstruction/mouse_gastrulation.csv",
    sep=";",
)
reconstruction_temp["data"] = "gastrulation"
reconstruction_df = pd.concat([reconstruction_df, reconstruction_temp], axis=0)
reconstruction_temp = pd.read_csv(
    "results/analysis/performance_evaluation/reconstruction/human_brain.csv"
)
reconstruction_temp["data"] = "brain"
reconstruction_df = pd.concat([reconstruction_df, reconstruction_temp], axis=0)
reconstruction_temp = pd.read_csv(
    "results/analysis/performance_evaluation/reconstruction/scMM_brain_recon_performance.csv"
)
reconstruction_temp["data"] = "brain"
# sort the data columns by the order of reconstruction_df
reconstruction_temp = reconstruction_temp[reconstruction_df.columns]
reconstruction_df = pd.concat([reconstruction_df, reconstruction_temp], axis=0)
reconstruction_df = reconstruction_df.drop(columns=["random_seed", "binary_threshold"])
reconstruction_df.reset_index(drop=True, inplace=True)
reconstruction_df["model"] = [
    "MultiVI" if x == "multiVI" else x for x in reconstruction_df["model"].values
]
# set the order of the models to 'MultiVI', 'scMM', 'multiDGD'
reconstruction_df["model"] = reconstruction_df["model"].astype("category")
reconstruction_df["model"] = reconstruction_df["model"].cat.set_categories(
    ["MultiVI", "scMM", "multiDGD"]
)

###############
# reconstruction performance (RMSE, Accuracy, ...) of DGD (also make binary atac trained version) and MultiVI
###############
pointplot_scale = 0.5
pointplot_errwidth = 0.7
pointplot_capsize = 0.2
ax_list.append(plt.subplot(gs[0:3, 0:2]))
# label the first row as A
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

sns.pointplot(
    data=reconstruction_df,
    x="data",
    y="RMSE (rna)",
    hue="model",
    ax=ax_list[-1],
    palette=palette_3colrs,
    errorbar="se",
    dodge=0.3,
    markers=".",
    linestyles="",
    scale=pointplot_scale,
    errwidth=pointplot_errwidth,
    capsize=pointplot_capsize,
)

ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_xlabel("Dataset")
ax_list[-1].set_ylabel("Root-Mean-Square Error")
ax_list[-1].set_title("Reconstruction (RNA) \u2193")
ax_list[-1].legend().remove()

ax_list.append(plt.subplot(gs[0:3, 2:4]))
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
sns.pointplot(
    data=reconstruction_df,
    x="data",
    y="balanced accuracy",
    hue="model",
    ax=ax_list[-1],
    palette=palette_3colrs,
    errorbar="se",
    dodge=0.3,
    markers=".",
    linestyles="",
    scale=pointplot_scale,
    errwidth=pointplot_errwidth,
    capsize=pointplot_capsize,
)
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_xlabel("Dataset")
ax_list[-1].set_ylabel("Balanced accuracy")
ax_list[-1].set_title("Reconstruction (ATAC) \u2191")
ax_list[-1].legend(
    bbox_to_anchor=(1.05, 1.0), loc="upper left", frameon=False
).set_visible(False)

###############
# clustering performances fo DGD, MultiVI (and ?) on all 4 datasets
###############
ax_list.append(plt.subplot(gs[0:3, 4:6]))
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.0 + 2 * grid_letter_positions[1],
    "C",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
sns.pointplot(
    data=multimodel_clustering,
    x="data",
    y="ARI",
    hue="model",
    ax=ax_list[-1],
    palette=palette_models,
    errorbar="se",
    dodge=0.3,
    markers=".",
    linestyles="",
    scale=pointplot_scale,
    errwidth=pointplot_errwidth,
    capsize=pointplot_capsize,
)
ax_list[-1].set_ylim((0.37, 0.74))
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_ylabel("Adjusted Rand Index")
ax_list[-1].set_xlabel("Dataset")
ax_list[-1].set_title("Clustering \u2191")
ax_list[-1].legend(
    bbox_to_anchor=(1.5 + legend_x_dist, -0.3 + legend_y_dist),
    loc="upper left",
    frameon=False,
    title="model",
    alignment="left",
    ncol=2,
    columnspacing=0.3,
    handletextpad=handletextpad,
)

###############
# Batch effect removal metrics (ASW, ?)
###############
multimodel_clustering["ASW"] = [1 - x for x in multimodel_clustering["ASW"].values]
ax_list.append(plt.subplot(gs[0:3, 6:]))
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.0 + 2 * grid_letter_positions[1],
    "D",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
sns.pointplot(
    data=multimodel_clustering[multimodel_clustering["data"] != "brain"],
    x="data",
    y="ASW",
    hue="model",
    ax=ax_list[-1],
    palette=palette_2colrs,
    errorbar="se",
    dodge=0.3,
    markers=".",
    linestyles="",
    scale=pointplot_scale,
    errwidth=pointplot_errwidth,
    capsize=pointplot_capsize,
)
ax_list[-1].set_ylim((0.98, 1.11))
ax_list[-1].set_ylabel("1 - ASW")
ax_list[-1].set_xlabel("Dataset")
ax_list[-1].set_title("Batch effect removal \u2191")
ax_list[-1].legend(
    bbox_to_anchor=(-0.2 + legend_x_dist, -0.3 + legend_y_dist),
    loc="upper left",
    frameon=False,
    title="model",
    ncol=2,
    columnspacing=0.3,
    handletextpad=handletextpad,
).set_visible(False)
print("finished row 1: performance metrics")

####################################
####################################
# block lower left: example latent space visualization
####################################
####################################
# load model for this and next block
cluster_class_neworder, class_palette = make_palette_from_meta(data_name)
column_names = ["UMAP D1", "UMAP D2"]
if not os.path.exists("results/analysis/performance_evaluation/bonemarrow_umap.csv"):
    data_name = "human_bonemarrow"
    import anndata as ad

    adata = ad.read_h5ad("data/" + data_name + ".h5ad")
    train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
    trainset = adata[train_indices, :]
    model = DGD.load(
        data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
    )
    # get latent spaces in reduced dimensionality
    rep = model.representation.z.detach().numpy()
    correction_rep = model.correction_rep.z.detach().numpy()
    cell_labels = trainset.obs["cell_type"].values
    batch_labels = trainset.obs["Site"].values
    test_rep = model.test_rep.z.detach().numpy()

    # make umap
    n_neighbors = 50
    min_dist = 0.75
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist)
    projected = reducer.fit_transform(rep)
    plot_data = pd.DataFrame(projected, columns=column_names)
    plot_data["cell type"] = cell_labels
    plot_data["cell type"] = plot_data["cell type"].astype("category")
    plot_data["cell type"] = plot_data["cell type"].cat.set_categories(
        cluster_class_neworder
    )
    plot_data["batch"] = batch_labels
    plot_data["batch"] = plot_data["batch"].astype("category")
    plot_data["batch"] = plot_data["batch"].cat.set_categories(
        ["site1", "site2", "site3", "site4"]
    )
    plot_data["data set"] = "train"
    projected_test = reducer.transform(test_rep)
    plot_data_test = pd.DataFrame(projected_test, columns=column_names)
    plot_data_test["data set"] = "test"
    correction_df = pd.DataFrame(correction_rep, columns=["D1", "D2"])
    correction_df["batch"] = batch_labels
    correction_df["batch"] = correction_df["batch"].astype("category")
    correction_df["batch"] = correction_df["batch"].cat.set_categories(
        ["site1", "site2", "site3", "site4"]
    )
    train_test_df = plot_data.copy()
    train_test_df.drop(columns=["cell type", "batch"], inplace=True)
    train_test_df = pd.concat([train_test_df, plot_data_test], axis=0)
    train_test_df["data set"] = train_test_df["data set"].astype("category")
    train_test_df["data set"] = train_test_df["data set"].cat.set_categories(
        ["train", "test"]
    )
    # transform GMM means and save
    projected_gmm = reducer.transform(model.gmm.mean.detach().numpy())
    projected_gmm = pd.DataFrame(projected_gmm, columns=column_names)
    projected_gmm["type"] = "mean"
    gmm_samples = reducer.transform(model.gmm.sample(10000).detach().numpy())
    gmm_samples = pd.DataFrame(gmm_samples, columns=column_names)
    gmm_samples["type"] = "sample"
    projected_gmm = pd.concat([projected_gmm, gmm_samples], axis=0)
    # save files
    plot_data.to_csv(
        "results/analysis/performance_evaluation/bonemarrow_umap.csv", index=False
    )
    correction_df.to_csv(
        "results/analysis/performance_evaluation/bonemarrow_correction_umap.csv",
        index=False,
    )
    train_test_df.to_csv(
        "results/analysis/performance_evaluation/bonemarrow_train_test_umap.csv",
        index=False,
    )
    projected_gmm.to_csv(
        "results/analysis/performance_evaluation/bonemarrow_gmm_umap.csv", index=False
    )
else:
    plot_data = pd.read_csv(
        "results/analysis/performance_evaluation/bonemarrow_umap.csv"
    )
    plot_data["cell type"] = plot_data["cell type"].astype("category")
    plot_data["cell type"] = plot_data["cell type"].cat.set_categories(
        cluster_class_neworder
    )
    plot_data["batch"] = plot_data["batch"].astype("category")
    plot_data["batch"] = plot_data["batch"].cat.set_categories(
        ["site1", "site2", "site3", "site4"]
    )
    correction_df = pd.read_csv(
        "results/analysis/performance_evaluation/bonemarrow_correction_umap.csv"
    )
    correction_df["batch"] = correction_df["batch"].astype("category")
    correction_df["batch"] = correction_df["batch"].cat.set_categories(
        ["site1", "site2", "site3", "site4"]
    )
    train_test_df = pd.read_csv(
        "results/analysis/performance_evaluation/bonemarrow_train_test_umap.csv"
    )
    train_test_df["data set"] = train_test_df["data set"].astype("category")
    train_test_df["data set"] = train_test_df["data set"].cat.set_categories(
        ["train", "test"]
    )
    projected_gmm = pd.read_csv(
        "results/analysis/performance_evaluation/bonemarrow_gmm_umap.csv"
    )

ax_list.append(plt.subplot(gs[4:8, 0:3]))
# label the first row as B
ax_list[-1].text(
    grid_letter_positions[0] + 0.01,
    1.0 + grid_letter_positions[1] - 0.01,
    "E",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
sns.scatterplot(
    data=plot_data.sort_values(by="cell type"),
    x=column_names[0],
    y=column_names[1],
    hue="cell type",
    palette=class_palette,
    ax=ax_list[-1],
    s=point_size,
    alpha=alpha,
    linewidth=point_linewidth,
    rasterized=True,
)
# remove axis ticks
ax_list[-1].tick_params(
    axis="both", which="both", bottom=False, top=False, left=False, right=False
)
# also remove axis tick values
ax_list[-1].set_xticklabels([])
ax_list[-1].set_yticklabels([])
ax_list[-1].set_title("latent representation (train set)")
ax_list[-1].legend(
    bbox_to_anchor=(1.0 + legend_x_dist, 1.15 + legend_y_dist),
    loc="upper left",
    frameon=False,
    handletextpad=handletextpad * 2,
    markerscale=handlesize,
    ncol=1,
    title="bone marrow cell type",
    labelspacing=0.2,
)
for i in range(len(projected_gmm[projected_gmm["type"] == "mean"])):
    ax_list[-1].text(
        projected_gmm[projected_gmm["type"] == "mean"].iloc[i][column_names[0]],
        projected_gmm[projected_gmm["type"] == "mean"].iloc[i][column_names[1]],
        str(i),
        fontsize=6,
        color="black",
        ha="center",
        va="center",
        fontweight="bold",
        path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="w")],
    )

ax_list.append(plt.subplot(gs[9:, 0:2]))
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.0 + 2 * grid_letter_positions[1],
    "G",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
sns.scatterplot(
    data=correction_df.sort_values(by="batch"),
    x="D1",
    y="D2",
    hue="batch",
    palette=batch_palette,
    alpha=alpha,
    linewidth=point_linewidth,
    ax=ax_list[-1],
    s=point_size,
)
ax_list[-1].set_title("batch representation")
# move the legend to one row in the bottom
ax_list[-1].legend(
    bbox_to_anchor=(-0.5, -0.3),
    loc="upper left",
    alignment="left",
    frameon=False,
    handletextpad=handletextpad,
    markerscale=handlesize,
    columnspacing=0.1,
    title="batch",
    ncol=4,
)

ax_list[-1].set_xlabel(r"$Z^{cov}$ D1")
ax_list[-1].set_ylabel(r"$Z^{cov}$ D2")

####################################
####################################
# block lower right: cluster map
####################################
####################################
ax_list.append(plt.subplot(gs[4:, 5:]))
# label the first row as C
ax_list[-1].text(
    grid_letter_positions[0] - 0.35,
    1.0 + grid_letter_positions[1] - 0.03,
    "F",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
# compute clustering of trained latent space
if not os.path.exists(
    "results/analysis/performance_evaluation/bonemarrow_cluster_map.csv"
):
    data_name = "human_bonemarrow"
    import anndata as ad

    adata = ad.read_h5ad("data/" + data_name + ".h5ad")
    train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
    trainset = adata[train_indices, :]
    model = DGD.load(
        data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
    )
    df_relative_clustering = gmm_make_confusion_matrix(model)
    df_relative_clustering.to_csv(
        "results/analysis/performance_evaluation/bonemarrow_cluster_map.csv"
    )
else:
    df_relative_clustering = pd.read_csv(
        "results/analysis/performance_evaluation/bonemarrow_cluster_map.csv",
        index_col=0,
    )
if not os.path.exists(
    "results/analysis/performance_evaluation/bonemarrow_cluster_map_nonorm.csv"
):
    data_name = "human_bonemarrow"
    import anndata as ad

    adata = ad.read_h5ad("data/" + data_name + ".h5ad")
    train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
    trainset = adata[train_indices, :]
    model = DGD.load(
        data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
    )
    df_clustering = gmm_make_confusion_matrix(model, norm=False)
    df_clustering.to_csv(
        "results/analysis/performance_evaluation/bonemarrow_cluster_map_nonorm.csv"
    )
else:
    df_clustering = pd.read_csv(
        "results/analysis/performance_evaluation/bonemarrow_cluster_map_nonorm.csv",
        index_col=0,
    )

# prepare annotations (percentage of cluster represented by each cell type)
annotations = df_relative_clustering.to_numpy(dtype=np.float64).copy()
# reduce annotations for readability
annotations[annotations < 1] = None
df_relative_clustering = df_relative_clustering.fillna(0)
# choose color palette
cmap = sns.color_palette("GnBu", as_cmap=True)
# remove color bar
sns.heatmap(
    df_relative_clustering,
    annot=annotations,
    cmap=cmap,
    annot_kws={"size": heatmap_fontsize},
    cbar_kws={"shrink": 0.5, "location": "bottom"},
    xticklabels=True,
    yticklabels=True,
    mask=np.isnan(annotations),
    ax=ax_list[-1],
    alpha=0.8,
)
cbar = ax_list[-1].collections[0].colorbar
cbar.remove()
# rotate x labels and enforce every label to be shown
ax_list[-1].tick_params(axis="x", rotation=90, labelsize=heatmap_fontsize)
ax_list[-1].set_ylabel("Cell type")
ax_list[-1].set_xlabel("GMM component ID")
ax_list[-1].set_title("percentage of cell type in GMM cluster")

print("finished block 3: cluster map")

####################################
####################################
# last subplot: data efficiency on human bonemarrow
####################################
####################################
ax_list.append(plt.subplot(gs[9:, 2:4]))
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.0 + 2 * grid_letter_positions[1],
    "H",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
print("getting mouse correction rep")
if not os.path.exists(
    "results/analysis/performance_evaluation/gastrulation_correction.csv"
):
    data_name = "mouse_gastrulation"
    import mudata as md

    mudata = md.load_data("data/" + data_name + ".h5mu")
    train_indices = list(np.where(mudata.obs["train_val_test"] == "train")[0])
    trainset = mudata[train_indices, :]
    mudata = None
    model = DGD.load(
        data=trainset,
        save_dir=save_dir + "mouse_gastrulation/",
        model_name="mouse_gast_l20_h2-2_rs0",
    )
    # get latent spaces in reduced dimensionality
    correction_rep = model.correction_rep.z.detach().numpy()
    batch_labels = trainset.obs["stage"].values
    correction_df = pd.DataFrame(correction_rep, columns=["D1", "D2"])
    correction_df["batch"] = batch_labels
    correction_df["batch"] = correction_df["batch"].astype("category")
    correction_df.to_csv(
        "results/analysis/performance_evaluation/gastrulation_correction.csv",
        index=False,
    )
else:
    correction_df = pd.read_csv(
        "results/analysis/performance_evaluation/gastrulation_correction.csv"
    )
    correction_df["batch"] = correction_df["batch"].astype("category")
# batch_palette = ['palegoldenrod', 'cornflowerblue', 'coral', 'darkmagenta', 'darkslategray']
sns.scatterplot(
    data=correction_df.sort_values(by="batch"),
    x="D1",
    y="D2",
    hue="batch",
    palette="magma_r",  # palette=batch_palette,
    ax=ax_list[-1],
    s=point_size,
    alpha=alpha,
    linewidth=point_linewidth,
)
ax_list[-1].set_title("gastrulation stage rep.")

ax_list[-1].legend(
    bbox_to_anchor=(1.02 + legend_x_dist, 1.0 + legend_y_dist),
    loc="upper left",
    frameon=False,
    markerscale=handlesize,
    handletextpad=handletextpad,
    title="stage",
    ncol=1,
)
ax_list[-1].set_xlabel(r"$Z^{cov}$ D1")
ax_list[-1].set_ylabel(r"$Z^{cov}$ D2")

####################################
####################################
# save figure
####################################
####################################

plt.savefig(
    "results/analysis/plots/performance_evaluation/fig2_v5.png",
    dpi=300,
    bbox_inches="tight",
)
