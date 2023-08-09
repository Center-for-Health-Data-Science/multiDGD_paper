"""
Explore what visually effective things to show at the end of Fig 2

I am thinking of comparing PCAs and umaps with same parameters for best models of DGD and multiVI
to show more complex latent space

I still think it would be good to compute structure preservation
"""

import os
import pandas as pd
from omicsdgd import DGD
import numpy as np
import umap.umap_ as umap
import mudata as md
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as PathEffects

from omicsdgd.functions._data_manipulation import (
    load_testdata_as_anndata,
    load_data_from_name,
)
from omicsdgd.functions._analysis import (
    make_palette_from_meta,
    gmm_make_confusion_matrix,
)

# print(plt.rcParams.keys())
# exit()

save_dir = "../results/trained_models/"
data_dir = "../../data/"
metric_dir = "../results/analysis/performance_evaluation/"

# set up figure
figure_height = 8
n_cols = 3
n_rows = 1
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=0.7, hspace=0.0)
ax_list = []
# palette_2colrs = ['palegoldenrod', 'cornflowerblue']
palette_2colrs = ["#DAA327", "cornflowerblue"]
batch_palette = ["#EEE7A8", "cornflowerblue", "darkmagenta", "darkslategray"]
extra_palette = ["gray", "darkslategray", "#EEE7A8", "#BDE1CD"]
plt.rcParams.update(
    {
        "font.size": 4,
        "axes.linewidth": 0.3,
        "xtick.major.size": 1.5,
        "xtick.major.width": 0.3,
        "ytick.major.size": 1.5,
        "ytick.major.width": 0.3,
    }
)
handletextpad = 0.1
legend_x_dist, legend_y_dist = 0.0, 0.0
grid_letter_positions = [-0.1, 0.05]
grid_letter_fontsize = 8
grid_letter_fontfamily = "sans-serif"
grid_letter_fontweight = "bold"
heatmap_fontsize = 3
point_size = 0.2
linewidth = 0.5
alpha = 1
point_linewidth = 0.0
handlesize = 0.3

# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)

####################################
####################################
# block lower right: cluster map
####################################
####################################

data_name = "mouse_gastrulation"
model_name = "mouse_gast_l20_h2-2_rs0"


ax_list.append(plt.subplot(gs[0, 0:2]))
# label the first row as C
ax_list[-1].text(
    grid_letter_positions[0] - 0.3,
    1.0 + grid_letter_positions[1],
    "A",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
# compute clustering of trained latent space
if not os.path.exists(metric_dir + data_name + "_cluster_map.csv"):
    data = md.read(data_dir + data_name + ".h5mu", backed=False)
    train_indices = list(np.where(data.obs["train_val_test"] == "train")[0])
    trainset = data[train_indices, :]
    # is_train_df = pd.read_csv(data_dir + data_name + "/train_val_test_split.csv")
    # trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
    model = DGD.load(
        data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
    )
    df_relative_clustering = gmm_make_confusion_matrix(model)
    df_relative_clustering.to_csv(metric_dir + data_name + "_cluster_map.csv")
else:
    df_relative_clustering = pd.read_csv(
        metric_dir + data_name + "_cluster_map.csv", index_col=0
    )
if not os.path.exists(metric_dir + data_name + "_cluster_map_nonorm.csv"):
    data = md.read(data_dir + data_name + ".h5mu", backed=False)
    train_indices = list(np.where(data.obs["train_val_test"] == "train")[0])
    trainset = data[train_indices, :]
    model = DGD.load(
        data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
    )
    df_clustering = gmm_make_confusion_matrix(model, norm=False)
    df_clustering.to_csv(metric_dir + data_name + "_cluster_map_nonorm.csv")
else:
    df_clustering = pd.read_csv(
        metric_dir + data_name + "_cluster_map_nonorm.csv", index_col=0
    )
# """ version 1: heatmap with annotation
# prepare annotations (percentage of cluster represented by each cell type)
annotations = df_relative_clustering.to_numpy(dtype=np.float64).copy()
# reduce annotations for readability
annotations[annotations < 1] = None
df_relative_clustering = df_relative_clustering.fillna(0)
# choose color palette
cmap = sns.color_palette("GnBu", as_cmap=True)
sns.heatmap(
    df_relative_clustering,
    annot=annotations,
    # sns.heatmap(df_clustering, annot=annotations,
    cmap=cmap,
    annot_kws={"size": heatmap_fontsize},
    mask=np.isnan(annotations),
    ax=ax_list[-1],
    alpha=0.8,
)
ylabels = [
    df_clustering.index[x] + " (" + str(int(df_clustering.sum(axis=1)[x])) + ")"
    for x in range(df_clustering.shape[0])
]
ax_list[-1].set(yticklabels=ylabels)
ax_list[-1].tick_params(axis="x", rotation=90)
ax_list[-1].set_ylabel("Cell type")
ax_list[-1].set_xlabel("GMM component ID")
ax_list[-1].set_title(
    "percentage of cell type represented by GMM cluster (" + data_name + ")"
)
# """


data_name = "human_brain"
model_name = "human_brain_l20_h2-2_a2_long"

import mudata as md
import anndata as ad
import scipy
import torch


def load_testdata_as_anndata(name, dir_prefix=data_dir):
    """
    takes a dataname and returns the data's train and external test sets as anndata objects,
    as well as the modality switch and the library of the test set
    """

    data = load_data_from_name(name)

    if type(data) is md.MuData:
        # transform to anndata
        modality_switch = data["rna"].X.shape[1]
        adata = ad.AnnData(scipy.sparse.hstack((data["rna"].X, data["atac"].X)))
        adata.obs = data["rna"].obs
        adata.var = pd.DataFrame(
            index=data["rna"].var_names.tolist() + data["atac"].var_names.tolist(),
            data={
                "name": data["rna"].var["name"].values.tolist()
                + data["atac"].var["name"].values.tolist(),
                "feature_types": ["rna"] * modality_switch
                + ["atac"] * (adata.shape[1] - modality_switch),
            },
        )
        # adata.var['feature_types'] = 'atac'
        # adata.var['feature_types'][:modality_switch] = 'rna'
        data = None
        data = adata
    else:
        if hasattr(data, "layers"):
            data.X = data.layers["counts"]
        modality_switch = np.where(
            data.var["feature_types"].values != data.var["feature_types"].values[0]
        )[0][0]

    # make train and test subsets
    is_train_df = pd.read_csv(dir_prefix + name + "/train_val_test_split.csv")
    train_indices = is_train_df[is_train_df["is_train"] == "train"]["num_idx"].values
    test_indices = is_train_df[is_train_df["is_train"] == "iid_holdout"][
        "num_idx"
    ].values
    if not isinstance(
        data.X, scipy.sparse.csc_matrix
    ):  # type(data.X) is not scipy.sparse._csc.csc_matrix:
        data.X = data.X.tocsr()
    trainset = data.copy()[train_indices]
    testset = data.copy()[test_indices]

    library = torch.cat(
        (
            torch.sum(
                torch.Tensor(testset.X.todense())[:, :modality_switch], dim=-1
            ).unsqueeze(1),
            torch.sum(
                torch.Tensor(testset.X.todense())[:, modality_switch:], dim=-1
            ).unsqueeze(1),
        ),
        dim=1,
    )

    return trainset, testset, modality_switch, library


ax_list.append(plt.subplot(gs[0, 2]))
# label the first row as C
ax_list[-1].text(
    grid_letter_positions[0] - 0.5,
    1.0 + grid_letter_positions[1],
    "B",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
# compute clustering of trained latent space
if not os.path.exists(metric_dir + data_name + "_cluster_map.csv"):
    data = md.read(data_dir + data_name + ".h5mu", backed=False)
    train_indices = list(np.where(data.obs["train_val_test"] == "train")[0])
    trainset = data[train_indices, :]
    # trainset.obs["celltype"] = trainset.obs["atac_celltype"]
    model = DGD.load(
        data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
    )
    df_relative_clustering = gmm_make_confusion_matrix(model)
    df_relative_clustering.to_csv(metric_dir + data_name + "_cluster_map.csv")
else:
    df_relative_clustering = pd.read_csv(
        metric_dir + data_name + "_cluster_map.csv", index_col=0
    )
if not os.path.exists(metric_dir + data_name + "_cluster_map_nonorm.csv"):
    data = md.read(data_dir + data_name + ".h5mu", backed=False)
    train_indices = list(np.where(data.obs["train_val_test"] == "train")[0])
    trainset = data[train_indices, :]
    model = DGD.load(
        data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
    )
    df_clustering = gmm_make_confusion_matrix(model, norm=False)
    df_clustering.to_csv(metric_dir + data_name + "_cluster_map_nonorm.csv")
else:
    df_clustering = pd.read_csv(
        metric_dir + data_name + "_cluster_map_nonorm.csv", index_col=0
    )
# """ version 1: heatmap with annotation
# prepare annotations (percentage of cluster represented by each cell type)
annotations = df_relative_clustering.to_numpy(dtype=np.float64).copy()
# reduce annotations for readability
annotations[annotations < 1] = None
df_relative_clustering = df_relative_clustering.fillna(0)
# choose color palette
cmap = sns.color_palette("GnBu", as_cmap=True)
sns.heatmap(
    df_relative_clustering,
    annot=annotations,
    # sns.heatmap(df_clustering, annot=annotations,
    cmap=cmap,
    annot_kws={"size": heatmap_fontsize},
    mask=np.isnan(annotations),
    ax=ax_list[-1],
    alpha=0.8,
)
ylabels = [
    df_clustering.index[x] + " (" + str(int(df_clustering.sum(axis=1)[x])) + ")"
    for x in range(df_clustering.shape[0])
]
ax_list[-1].set(yticklabels=ylabels)
# rotate y labels
plt.setp(ax_list[-1].get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
ax_list[-1].tick_params(axis="x", rotation=90)
ax_list[-1].set_ylabel("Cell type")
ax_list[-1].set_xlabel("GMM component ID")
ax_list[-1].set_title(
    "percentage of cell type represented by GMM cluster (" + data_name + ")"
)


plt.savefig(
    "../results/analysis/plots/supplementaries/fig_supp_heatmaps.png",
    dpi=300,
    bbox_inches="tight",
)
