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

save_dir = "../results/trained_models/"
metric_dir = "../results/analysis/performance_evaluation/"
data_dir = "../../data/"
data_name = "human_bonemarrow"
model_name = "human_bonemarrow_l20_h2-3_test10e"  # this is just a speficiation of the model after test set inference

# set up figure
figure_height = 18
n_cols = 8
n_rows = 14
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
#gs_bottom = gridspec.GridSpec(4, n_cols)
gs.update(wspace=6.0, hspace=100.0)
#gs_bottom.update(wspace=6.0, hspace=200.0)
ax_list = []
palette_2colrs = ["#DAA327", "#015799"]
palette_models = ["#DAA327", "#BDE1CD", "palegoldenrod", "#015799"]
palette_3colrs = ["#DAA327", "#BDE1CD", "#015799"]
batch_palette = ["#EEE7A8", "cornflowerblue", "darkmagenta", "darkslategray"]
extra_palette = ["gray", "darkslategray", "#EEE7A8", "#BDE1CD"]
plt.rcParams.update(
    {
        "font.size": 6,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
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
strip_color = "black"
strip_size = 1
strip_alpha = 1
strip_line = 0.0

# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)

####################################
####################################
# row 1: performance metrics (comparison to multiVI)
####################################
####################################
# load needed analysis data

###
# load and combine reconstruction performances of all datasets (for MultiVI and multiDGD)
###
reconstruction_temp = pd.read_csv(metric_dir + "reconstruction/human_bonemarrow.csv")
reconstruction_df = reconstruction_temp
reconstruction_temp["data"] = "marrow"
reconstruction_temp = pd.read_csv(metric_dir + "reconstruction/mouse_gastrulation.csv")
reconstruction_temp["data"] = "gastrulation"
reconstruction_df = pd.concat([reconstruction_df, reconstruction_temp], axis=0)
reconstruction_temp = pd.read_csv(metric_dir + "reconstruction/human_brain.csv")
reconstruction_temp["data"] = "brain"
reconstruction_df = pd.concat([reconstruction_df, reconstruction_temp], axis=0)
reconstruction_df["Model"] = [x if x == "multiDGD" else "MultiVI" for x in reconstruction_df["model"].values]
reconstruction_df["Data"] = reconstruction_df["data"]

# extract clustering information and shorten the original df
# make clustering df from relevant columns of reconstruction_df
clustering_df = reconstruction_df[["Data", "Model", "random_seed", "ARI", "ASW"]].copy()

reconstruction_df = reconstruction_df[["RMSE (rna)", "balanced accuracy", "Model", "Data", "random_seed"]]

# add the new AUPRC metric
random_seeds = [0, 37, 8790]*3
data_names = ["human_bonemarrow"]*3 + ["mouse_gastrulation"]*3 + ["human_brain"]*3
data_names_plot = ["marrow"]*3 + ["gastrulation"]*3 + ["brain"]*3
n_components = [22, 22, 22, 37, 37, 37, 16, 16, 16]
for i,data_name in enumerate(data_names):
    result_path = "../results/revision/analysis/batch_integration/" + data_name + "/"
    plot_path = "../results/revision/plots/" + data_name + "/"

    # first load multiDGD results
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "__ncomp"
        + str(n_components[i])
        + "_auprc_spearman.csv"
    )
    temp_df["Model"] = "multiDGD"
    temp_df["random_seed"] = random_seeds[i]
    temp_df["Data"] = data_names_plot[i]

    if i == 0:
        df_a_s = temp_df
    else:
        df_a_s = pd.concat([df_a_s, temp_df])
    
    # load MultiVI results
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_auprc_spearman_mvi.csv"
    )
    temp_df["Model"] = "MultiVI"
    temp_df["random_seed"] = random_seeds[i]
    temp_df["Data"] = data_names_plot[i]

    df_a_s = pd.concat([df_a_s, temp_df])

# get the auprc value from df_a_s by model, data and random seed
reconstruction_df["AUPRC"] = [df_a_s[(df_a_s["Model"] == x) & (df_a_s["Data"] == y) & (df_a_s["random_seed"] == z)]["auprc"].values[0] for x,y,z in zip(reconstruction_df["Model"], reconstruction_df["Data"], reconstruction_df["random_seed"])]

# keep only what we need
reconstruction_df = reconstruction_df[["RMSE (rna)", "Model", "Data", "random_seed", "AUPRC"]]

# also read scMM
temp_df = pd.read_csv("../results/revision/analysis/performance/scMM_brain_recon_performance_revision.csv")
temp_df["Model"] = "scMM"
temp_df["Data"] = "brain"
temp_df = temp_df[["RMSE (rna)", "Model", "Data", "random_seed", "AUPRC"]]
reconstruction_df = pd.concat([reconstruction_df, temp_df], axis=0)

# set the order of the models to 'MultiVI', 'scMM', 'multiDGD'
reconstruction_df["Model"] = reconstruction_df["Model"].astype("category")
reconstruction_df["Model"] = reconstruction_df["Model"].cat.set_categories(
    ["MultiVI", "scMM", "multiDGD"]
)

###
# now add the clustering results from cobolt and scmm
###
clustering_cobolt = pd.read_csv(metric_dir + "cobolt_human_brain_aris.csv")
clustering_cobolt["Data"] = "brain"
clustering_cobolt["Model"] = "Cobolt"
clustering_cobolt = clustering_cobolt[["Data", "Model", "random_seed", "ARI"]]
clustering_scmm = pd.read_csv(metric_dir + "scmm_human_brain_aris.csv")
clustering_scmm["Data"] = "brain"
clustering_scmm["Model"] = "scMM"
clustering_scmm = clustering_scmm[["Data", "Model", "random_seed", "ARI"]]
multimodel_clustering = pd.concat(
    [clustering_df, clustering_cobolt, clustering_scmm], axis=0
)
multimodel_clustering["Model"] = multimodel_clustering["Model"].astype("category")
multimodel_clustering["Model"] = multimodel_clustering["Model"].cat.set_categories(
    ["MultiVI", "scMM", "Cobolt", "multiDGD"]
)
# clustering_df["ARI"] = clustering_df["ARI"].round(2)

###############
# reconstruction performance (RMSE, Accuracy, ...) of DGD (also make binary atac trained version) and MultiVI
###############
pointplot_scale = 0.5
pointplot_errwidth = 0.7
pointplot_capsize = 0.2
ax_list.append(plt.subplot(gs[0:3, 0:2]))
# label the first row as A

ax_list[-1].text(
    2.55,
    1.3,
    "Performance benchmark",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)

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
    x="Data",
    y="RMSE (rna)",
    hue="Model",
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
sns.stripplot(
    data=reconstruction_df,
    x="Data",
    y="RMSE (rna)",
    hue="Model",
    #palette=palette_3colrs,
    color=strip_color,
    ax=ax_list[-1],
    dodge=True,
    size=strip_size,
    alpha=strip_alpha,
    linewidth=strip_line,
    legend=False,
)

ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_xlabel("Dataset")
ax_list[-1].set_ylabel("RMSE")
ax_list[-1].set_title("Reconstruction error (RNA) \u2193")
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
    x="Data",
    y="AUPRC",
    hue="Model",
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
sns.stripplot(
    data=reconstruction_df,
    x="Data",
    y="AUPRC",
    hue="Model",
    ax=ax_list[-1],
    #palette=palette_3colrs,
    color=strip_color,
    dodge=True,
    size=strip_size,
    alpha=strip_alpha,
    linewidth=strip_line,
    legend=False,
)
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_xlabel("Dataset")
ax_list[-1].set_ylabel("AUPRC")
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
    x="Data",
    y="ARI",
    hue="Model",
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
sns.stripplot(
    data=multimodel_clustering,
    x="Data",
    y="ARI",
    hue="Model",
    #palette=palette_models,
    color=strip_color,
    ax=ax_list[-1],
    dodge=True,
    size=strip_size,
    alpha=strip_alpha,
    linewidth=strip_line,
    legend=False,
)
ax_list[-1].set_ylim((0.37, 0.74))
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_ylabel("Adjusted Rand Index")
ax_list[-1].set_xlabel("Dataset")
ax_list[-1].set_title("Clustering \u2191")
ax_list[-1].legend(
    bbox_to_anchor=(1.5 + legend_x_dist, -0.3 + legend_y_dist),
    #bbox_to_anchor=(1.5 + legend_x_dist, -0.25 + legend_y_dist),
    loc="upper left",
    frameon=False,
    title="Model",
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
    data=multimodel_clustering[multimodel_clustering["Data"] != "brain"],
    x="Data",
    y="ASW",
    hue="Model",
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
sns.stripplot(
    data=multimodel_clustering[multimodel_clustering["Data"] != "brain"],
    x="Data",
    y="ASW",
    hue="Model",
    #palette=palette_2colrs,
    color=strip_color,
    ax=ax_list[-1],
    dodge=True,
    size=strip_size,
    alpha=strip_alpha,
    linewidth=strip_line,
    legend=False,
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
data_name = "human_bonemarrow"
cluster_class_neworder, class_palette = make_palette_from_meta(data_name)
column_names = ["UMAP D1", "UMAP D2"]
if not os.path.exists(metric_dir + "bonemarrow_umap.csv"):
    data_name = "human_bonemarrow"
    import anndata as ad

    adata = ad.read_h5ad(data_dir + data_name + ".h5ad")
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
    plot_data.to_csv(metric_dir + "bonemarrow_umap.csv", index=False)
    correction_df.to_csv(
        metric_dir + "bonemarrow_correction_umap.csv",
        index=False,
    )
    train_test_df.to_csv(
        metric_dir + "bonemarrow_train_test_umap.csv",
        index=False,
    )
    projected_gmm.to_csv(metric_dir + "bonemarrow_gmm_umap.csv", index=False)
else:
    plot_data = pd.read_csv(metric_dir + "bonemarrow_umap.csv")
    plot_data["cell type"] = plot_data["cell type"].astype("category")
    plot_data["cell type"] = plot_data["cell type"].cat.set_categories(
        cluster_class_neworder
    )
    plot_data["batch"] = plot_data["batch"].astype("category")
    plot_data["batch"] = plot_data["batch"].cat.set_categories(
        ["site1", "site2", "site3", "site4"]
    )
    correction_df = pd.read_csv(metric_dir + "bonemarrow_correction_umap.csv")
    correction_df["batch"] = correction_df["batch"].astype("category")
    correction_df["batch"] = correction_df["batch"].cat.set_categories(
        ["site1", "site2", "site3", "site4"]
    )
    train_test_df = pd.read_csv(metric_dir + "bonemarrow_train_test_umap.csv")
    train_test_df["data set"] = train_test_df["data set"].astype("category")
    train_test_df["data set"] = train_test_df["data set"].cat.set_categories(
        ["train", "test"]
    )
    projected_gmm = pd.read_csv(metric_dir + "bonemarrow_gmm_umap.csv")

ax_list.append(plt.subplot(gs[10:, 4:7]))
#ax_list.append(plt.subplot(gs_bottom[0:, 4:7]))
# label the first row as B
ax_list[-1].text(
    grid_letter_positions[0] + 0.01,
    1.0 + grid_letter_positions[1] + 0.02,
    "J",
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
ax_list[-1].set_title("basal representation (marrow, train)")
ax_list[-1].legend(
    bbox_to_anchor=(1.02 + legend_x_dist, 1.6 + legend_y_dist),
    loc="upper left",
    frameon=False,
    handletextpad=handletextpad * 2,
    markerscale=handlesize,
    ncol=1,
    title="Cell type",
    labelspacing=0.2,
)
#"""
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
#"""

ax_list.append(plt.subplot(gs[10:13, 2:4]))
#ax_list.append(plt.subplot(gs_bottom[0:3, 2:4]))

ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.0 + 2 * grid_letter_positions[1],
    "I",
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
ax_list[-1].set_title("batch (marrow)")
# move the legend to one row in the bottom
ax_list[-1].legend(
    bbox_to_anchor=(-0.3, -0.3),
    loc="upper left",
    alignment="left",
    frameon=False,
    handletextpad=handletextpad,
    markerscale=handlesize,
    columnspacing=0.1,
    title="batch",
    ncol=2,
)

ax_list[-1].set_xlabel(r"$Z^{cov}$ D1")
ax_list[-1].set_ylabel(r"$Z^{cov}$ D2")

####################################
####################################
# gastrulation
####################################
####################################
ax_list.append(plt.subplot(gs[10:13, 0:2]))
#ax_list.append(plt.subplot(gs_bottom[0:3, 0:2]))

ax_list[-1].text(
    2.35,
    1.35,
    "Disentangled representations",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)

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
if not os.path.exists(metric_dir + "gastrulation_correction.csv"):
    data_name = "mouse_gastrulation"
    import mudata as md

    mudata = md.read(data_dir + data_name + ".h5mu", backed=False)
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
        metric_dir + "gastrulation_correction.csv",
        index=False,
    )
else:
    correction_df = pd.read_csv(metric_dir + "gastrulation_correction.csv")
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
ax_list[-1].set_title("stage (gastrulation)")
"""
ax_list[-1].legend(
    bbox_to_anchor=(1.02 + legend_x_dist, 1.0 + legend_y_dist),
    loc="upper left",
    frameon=False,
    markerscale=handlesize,
    handletextpad=handletextpad,
    title="stage",
    ncol=1,
)
"""
ax_list[-1].legend(
    bbox_to_anchor=(-0.3, -0.3),
    loc="upper left",
    alignment="left",
    frameon=False,
    handletextpad=handletextpad,
    markerscale=handlesize,
    columnspacing=0.1,
    title="stage",
    ncol=3,
)
ax_list[-1].set_xlabel(r"$Z^{cov}$ D1")
ax_list[-1].set_ylabel(r"$Z^{cov}$ D2")

####################################
####################################
# moved and new stuff
####################################
####################################

# add modality integration
"""
ax_list.append(plt.subplot(gs[4:8, 4:]))

umap_df = pd.read_csv("../results/analysis/modality_integration/human_bonemarrow_l20_h2-3_rs0_unpaired0percent_latent_integration_umap_all.csv")
umap_df['data_set'] = umap_df['data_set'].astype('category')
umap_df['data_set'] = umap_df['data_set'].cat.set_categories(['train', 'test (paired)', 'test (atac)', 'test (rna)'])
sns.scatterplot(data=umap_df,#.sort_values(by='data_set'), 
                x='UMAP1', y='UMAP2', hue='data_set', palette=extra_palette,
                s=point_size, ax=ax_list[-1], alpha=0.5, linewidth=point_linewidth)
# move the legend to one row in the bottom
ax_list[-1].legend(bbox_to_anchor=(1.0+legend_x_dist, 1.+legend_y_dist),
                    loc='upper left',
                    frameon=False,
                    handletextpad=handletextpad*2,
                    title='data set',
                    markerscale=handlesize)
                    # minimize distances between handles and labels and between items
                    #labelspacing=0.1)
ax_list[-1].set_title("Integration of single modalities")
# remove axis ticks
ax_list[-1].tick_params(
    axis="both", which="both", bottom=False, top=False, left=False, right=False
)
# also remove axis tick values
ax_list[-1].set_xticklabels([])
ax_list[-1].set_yticklabels([])
"""

# add data efficiency
ax_list.append(plt.subplot(gs[5:8, 0:4]))
ax_list[-1].text(
    1.05,
    1.35,
    "Data efficiency",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
ax_list[-1].text(
    grid_letter_positions[0] * 0.7,
    1.05 + 2 * grid_letter_positions[1],
    "E",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)

# add a horizontal line at 1
ax_list[-1].axhline(1, color='grey', linestyle='--', linewidth=0.5)

# x axis: percentage of data set
efficiency_df_mvi = pd.read_csv('../results/analysis/performance_evaluation/human_bonemarrow_data_efficiency_mvi.csv')
# change multiVI to MultiVI in model column
efficiency_df_mvi['model'] = ['MultiVI' if x == 'multiVI' else x for x in efficiency_df_mvi['model'].values]
efficiency_df_mvi['fraction'] = efficiency_df_mvi['fraction'] * 100
loss_ratios = []
for count, loss in enumerate(efficiency_df_mvi['loss'].values):
    rs = efficiency_df_mvi['random_seed'].values[count]
    base_loss = efficiency_df_mvi[(efficiency_df_mvi['random_seed'] == rs) & (efficiency_df_mvi['fraction'] == 100)]['loss'].item()
    loss_ratios.append(loss / base_loss)
efficiency_df_mvi['loss ratio'] = loss_ratios
efficiency_df = pd.read_csv('../results/analysis/performance_evaluation/human_bonemarrow_data_efficiency.csv', sep=';')
efficiency_df['fraction'] = efficiency_df['fraction'] * 100
loss_ratios = []
for count, loss in enumerate(efficiency_df['loss'].values):
    rs = efficiency_df['random_seed'].values[count]
    base_loss = efficiency_df[(efficiency_df['random_seed'] == rs) & (efficiency_df['fraction'] == 100)]['loss'].item()
    loss_ratios.append(loss / base_loss)
efficiency_df['loss ratio'] = loss_ratios
efficiency_df = pd.concat([efficiency_df_mvi, efficiency_df], axis=0)
# make fraction column integers
efficiency_df['fraction'] = efficiency_df['fraction'].astype(int)
efficiency_df['n_samples_k'] = [str(round(x/1000, 1))+'k' for x in efficiency_df['n_samples'].values]

sns.pointplot(data=efficiency_df, x='fraction', y='loss ratio',
             hue='model', palette=palette_2colrs, ax=ax_list[-1],
             errorbar="se",markers=".",scale=pointplot_scale,
             errwidth=pointplot_errwidth,capsize=pointplot_capsize,
             linestyles=["", ""]
             )
def percent_to_n(x):
    print(x)
    print(type(x))
    return int(x * 0.01 * 56714)
def n_to_percent(x):
    return int(x / 56714 * 100)
#secax = ax_list[-1].secondary_xaxis('top', functions=(percent_to_n, n_to_percent))
#secax.set_xlabel('Number of training samples')
ax2 = ax_list[-1].twiny()
sns.pointplot(data=efficiency_df.sort_values(by='n_samples'), x='n_samples_k', y='loss ratio',
             hue='model', palette=palette_2colrs, ax=ax2,
             errorbar="se",markers=".",scale=pointplot_scale,
             errwidth=pointplot_errwidth,capsize=pointplot_capsize,
             linestyles=["", ""]
             )
sns.stripplot(
    data=efficiency_df,
    x='fraction',
    y='loss ratio',
    hue='model',
    #palette=palette_2colrs,
    color=strip_color,
    ax=ax_list[-1],
    dodge=True,
    size=strip_size,
    alpha=strip_alpha,
    linewidth=strip_line,
    legend=False
)
ax2.legend_.remove()
ax_list[-1].legend(bbox_to_anchor=(1.0+legend_x_dist, 1.+legend_y_dist),
                   loc='upper left', frameon=False,title='model',
                   markerscale=handlesize*3,
                   handletextpad=handletextpad*2).set_visible(False)
#ax_list[-1].text(30, 1.01, 'placeholder', fontdict={'color': 'red'})
ax_list[-1].set_xlabel('Percentage of training set (marrow)')
ax2.set_xlabel('Number of training samples')


mouse_df = pd.read_csv('../results/revision/analysis/efficiency/mouse_gast_multidgd_5percent.csv')
mouse_df["Model"] = "multiDGD"
mouse_df["Feature selection"] = "in 5%"
temp_df = pd.read_csv('../results/revision/analysis/efficiency/mouse_gast_multidgd_full.csv')
temp_df["Model"] = "multiDGD"
temp_df["Feature selection"] = "all"
mouse_df = pd.concat([mouse_df, temp_df], axis=0)
temp_df = pd.read_csv('../results/revision/analysis/efficiency/mouse_gast_multivi_5percent.csv')
temp_df["Model"] = "MultiVI"
temp_df["Feature selection"] = "in 5%"
mouse_df = pd.concat([mouse_df, temp_df], axis=0)
temp_df = pd.read_csv('../results/revision/analysis/efficiency/mouse_gast_multivi_full_even.csv')
temp_df["Model"] = "MultiVI"
temp_df["Feature selection"] = "all"
mouse_df = pd.concat([mouse_df, temp_df], axis=0)
temp_df = pd.read_csv('../results/revision/analysis/efficiency/mouse_gast_multivi_full_odd.csv')
temp_df["Model"] = "MultiVI"
temp_df["Feature selection"] = "all"
mouse_df = pd.concat([mouse_df, temp_df], axis=0)
mouse_df["Model"] = mouse_df["Model"].astype("category")
mouse_df["Model"] = mouse_df["Model"].cat.set_categories(
    ["MultiVI", "multiDGD"]
)

"""
mouse_df = pd.DataFrame({
    'Model': ['MultiVI', 'MultiVI', 'multiDGD', 'multiDGD'],
    'Feature selection': ['in 5%', 'all']*2,
    "AUPRC": [0.3850507290738613, 0.3869164169579223, 0.3904613606402934, 0.39090892534633226],
    "AUPRC_se": [0.002130566293342677, 0.002983842605418741, 0.002142662011430513, 0.002145051801665028],
    "RMSE": [2.342944383621216, 2.4637396335601807, 1.714585781097412, 1.6939202547073364],
    "RMSE_se": [0.02120981365442276, 0.021737542003393173, 0.012759659439325333, 0.01272590272128582]
})
"""

ax_list.append(plt.subplot(gs[5:8, 4:6]))
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.05 + 2 * grid_letter_positions[1],
    "F",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
# make a pointplot of the RMSE with standard errors
sns.pointplot(data=mouse_df, x='Feature selection', y='rmse', hue='Model', ax=ax_list[-1],
              palette=palette_2colrs, markers=".", scale=pointplot_scale,
              errwidth=pointplot_errwidth, capsize=pointplot_capsize,
              linestyles=["", ""],errorbar="se", dodge=0.3
              )
# remove legend
ax_list[-1].legend_.remove()
ax_list[-1].set_xlabel('Feature selection (gastrulation)')
ax_list[-1].set_ylabel('RMSE')
ax_list[-1].set_title("Reconstruction error (RNA) \u2193")

ax_list.append(plt.subplot(gs[5:8, 6:]))
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.05 + 2 * grid_letter_positions[1],
    "G",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
#ax_list[-1].axhline(0.14943362590918774, color='grey', linestyle='--', linewidth=0.5) # baseline, fraction of non-zero values
# make a pointplot of the RMSE with standard errors
sns.pointplot(data=mouse_df, x='Feature selection', y='auprc', hue='Model', ax=ax_list[-1],
              palette=palette_2colrs, markers=".", scale=pointplot_scale,
              errwidth=pointplot_errwidth, capsize=pointplot_capsize,
              linestyles=["", ""],errorbar="se", dodge=0.3
              )
# remove legend
ax_list[-1].legend_.remove()
ax_list[-1].set_xlabel('Feature selection (gastrulation)')
ax_list[-1].set_ylabel('AUPRC')
ax_list[-1].set_title("Reconstruction (ATAC) \u2191")

####################################
####################################
# save figure
####################################
####################################

plt.savefig(
    "../results/revision/plots/main/fig2_revision_v4",
    dpi=300,
    bbox_inches="tight",
    format="pdf",
)
