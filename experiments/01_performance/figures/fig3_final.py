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
figure_height = 8
n_cols = 7
n_rows = 7
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
#gs_bottom = gridspec.GridSpec(4, n_cols)
gs.update(wspace=20.0, hspace=3.0)
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
        #"font.sans-serif": "Helvetica",
        "axes.linewidth": 0.3,
        "xtick.major.size": 1.5,
        "xtick.major.width": 0.3,
        "ytick.major.size": 1.5,
        "ytick.major.width": 0.3,
        #"lines.linewidth": 0.5,
    }
)
handletextpad = 0.1
legend_x_dist, legend_y_dist = -0.02, -0.0
grid_letter_positions = [-0.1, 0.05]
grid_letter_fontsize = 7
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

ax_list.append(plt.subplot(gs[0:, 2:6]))
#ax_list.append(plt.subplot(gs_bottom[0:, 4:7]))
# label the first row as B
ax_list[-1].text(
    grid_letter_positions[0] + 0.1,
    1.0 + grid_letter_positions[1] - 0.02,
    "C",
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
ax_list[-1].set_title("Basal representation (marrow, train)")
ax_list[-1].legend(
    #bbox_to_anchor=(1.02 + legend_x_dist, 1.6 + legend_y_dist),
    bbox_to_anchor=(1.02 + legend_x_dist, 1.0 + legend_y_dist),
    loc="upper left",
    alignment="left",
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

ax_list.append(plt.subplot(gs[0:3, 0:2]))
#ax_list.append(plt.subplot(gs_bottom[0:3, 2:4]))

ax_list[-1].text(
    grid_letter_positions[0] * 1.5,
    1.0 + 2 * grid_letter_positions[1],
    "A",
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
ax_list[-1].set_title("Batch (marrow)")
# move the legend to one row in the bottom
ax_list[-1].legend(
    #bbox_to_anchor=(-0.3, -0.3),
    bbox_to_anchor=(1.02 + legend_x_dist, 1.0 + legend_y_dist),
    loc="upper left",
    alignment="left",
    frameon=False,
    handletextpad=handletextpad,
    markerscale=handlesize,
    columnspacing=0.1,
    title="Batch",
    ncol=1,
)

ax_list[-1].set_xlabel(r"$Z^{cov}$ D1")
ax_list[-1].set_ylabel(r"$Z^{cov}$ D2")

####################################
####################################
# gastrulation
####################################
####################################
ax_list.append(plt.subplot(gs[4:, 0:2]))
#ax_list.append(plt.subplot(gs_bottom[0:3, 0:2]))

"""
ax_list[-1].text(
    2.35,
    1.25,
    "Disentangled representations",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
"""

ax_list[-1].text(
    grid_letter_positions[0] * 1.5,
    1.0 + 2 * grid_letter_positions[1],
    "B",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)

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
ax_list[-1].set_title("Stage (gastrulation)")
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
    #bbox_to_anchor=(-0.3, -0.3),
    bbox_to_anchor=(1.02 + legend_x_dist, 1.0 + legend_y_dist),
    loc="upper left",
    alignment="left",
    frameon=False,
    handletextpad=handletextpad,
    markerscale=handlesize,
    columnspacing=0.1,
    title="Stage",
    ncol=1,
)
ax_list[-1].set_xlabel(r"$Z^{cov}$ D1")
ax_list[-1].set_ylabel(r"$Z^{cov}$ D2")

plt.savefig(
    "../results/figures/fig3_final.pdf",
    dpi=300,
    bbox_inches="tight",
    format="pdf",
)
