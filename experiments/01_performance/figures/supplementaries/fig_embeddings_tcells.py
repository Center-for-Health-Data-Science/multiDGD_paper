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
from omicsdgd.functions._analysis import make_palette_from_meta

import anndata as ad
import scipy
import scvi

#####################
# define model names, directory and batches
#####################

save_dir = "../results/trained_models/"
data_dir = "../../data/"
metric_dir = "../results/analysis/performance_evaluation/"

####################################
# flexible parameters
####################################
figure_height = 3.5
n_cols = 2
n_rows = 1
grid_wspace = 1.5
grid_hspace = 0.5

####################################
# fixed figure design
####################################
# set up figure and grid
cm = 1 / 2.54
fig = plt.figure(figsize=(12 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=grid_wspace, hspace=grid_hspace)
ax_list = []
# fonts
# general text
text_size = 8
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
# grid letters
grid_letter_positions = [-0.1, 0.05]
grid_letter_fontsize = 8
grid_letter_fontfamily = "sans-serif"
grid_letter_fontweight = "bold"
point_size = 1
# colors
palette_2colrs = ["palegoldenrod", "cornflowerblue"]
batch_palette = ["palegoldenrod", "cornflowerblue", "darkmagenta", "darkslategray"]
palette_3colrs = [
    "lightgray",
    "cornflowerblue",
    "darkmagenta",
    "darkolivegreen",
    "firebrick",
    "midnightblue",
]
palette_continuous_1 = "GnBu"
palette_continuous_2 = "magma_r"
# legend set up
legend_x_dist, legend_y_dist = 0.02, 0.0
handletextpad = 0.1
# scatter plot
point_size = 0.3
alpha = 1
point_linewidth = 0.0
handlesize = 0.2
# line plot
linewidth = 0.5

# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)

#####################
# load data
#####################
# first load the data for labels
data_names = ["human_bonemarrow"]
dgd_names = ["human_bonemarrow_l20_h2-3"]
mvi_names = ["l20_e2_d2"]
dgd_neighbors = [15, 15, 15]
dgd_dists = [0.5, 0.5, 0.5]
mvi_neighbors = [15, 15, 15]
mvi_dists = [0.5, 0.5, 0.5]
dgd_grid_letters = ["A", "B", "C"]
mvi_grid_letters = ["D", "E", "F"]
plotting_key = ["cell_type"]
legendscale = 0.2
mvi_batch_keys = ["Site"]

#####################
# first row: DGD
#####################
print("doing DGD")

for i, data_name in enumerate(data_names):
    model_name = dgd_names[i]
    cluster_class_neworder, class_palette = make_palette_from_meta(data_name)
    column_names = ["UMAP D1", "UMAP D2"]
    if not os.path.exists(metric_dir + "dgd_" + data_name + "_TCELL_umap.csv"):
        print("making umap for data " + data_name)
        data = ad.read_h5ad(data_dir + "human_bonemarrow.h5ad")
        modality_switch = 13431
        train_indices = list(np.where(data.obs["train_val_test"] == "train")[0])
        trainset = data[train_indices, :]

        # subset the data to t cells
        # string that is present in all: '+ T'
        t_cell_string = "+ T"
        t_cell_indices = [
            i
            for i, s in enumerate(trainset.obs["cell_type"].values)
            if t_cell_string in s
        ]
        # np.where(np.char.find(trainset.obs['cell_type'].values, t_cell_string) != -1)[0]

        model = DGD.load(
            data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
        )
        # get latent spaces in reduced dimensionality
        rep = model.representation.z.detach().numpy()[t_cell_indices, :]
        cell_labels = trainset.obs[plotting_key[i]].values[t_cell_indices]

        # make umap
        n_neighbors = dgd_neighbors[i]
        min_dist = dgd_dists[i]
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist)
        projected = reducer.fit_transform(rep)
        plot_data = pd.DataFrame(projected, columns=column_names)
        plot_data["cell type"] = cell_labels
        plot_data["cell type"] = plot_data["cell type"].astype("category")
        # plot_data['cell type'] = plot_data['cell type'].cat.set_categories(cluster_class_neworder)
        plot_data.to_csv(
            metric_dir + "dgd_" + data_name + "_TCELL_umap.csv", index=False
        )
    else:
        plot_data = pd.read_csv(metric_dir + "dgd_" + data_name + "_TCELL_umap.csv")
        plot_data["cell type"] = plot_data["cell type"].astype("category")
        plot_data["cell type"] = plot_data["cell type"].cat.set_categories(
            cluster_class_neworder
        )

    ax_list.append(plt.subplot(gs[0]))
    ax_list[-1].text(
        grid_letter_positions[0],
        1.0 + grid_letter_positions[1],
        "A",
        transform=ax_list[-1].transAxes + trans,
        fontsize=grid_letter_fontsize,
        va="bottom",
        fontfamily=grid_letter_fontfamily,
        fontweight=grid_letter_fontweight,
    )
    sns.scatterplot(
        data=plot_data,  # .sort_values(by='cell type'),
        x=column_names[0],
        y=column_names[1],
        hue="cell type",
        palette=class_palette,
        ax=ax_list[-1],
        s=point_size,
        alpha=alpha,
        linewidth=point_linewidth,
    )
    # legend
    # extract handles and labels and use the last 4
    handles, labels = ax_list[-1].get_legend_handles_labels()
    handles = handles[-4:]
    labels = labels[-4:]
    ax_list[-1].legend(
        bbox_to_anchor=(1.0 + legend_x_dist, 1.0 + legend_y_dist),
        handles=handles,
        labels=labels,
        title="bone marrow cell type\n(T cells)",
        loc="upper left",
        frameon=False,
        handletextpad=handletextpad,
        markerscale=handlesize,
    )
    ax_list[-1].set_title("multiDGD")
    # remove axis ticks
    ax_list[-1].tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False
    )
    # also remove axis tick values
    ax_list[-1].set_xticklabels([])
    ax_list[-1].set_yticklabels([])

###################
# second row: MVI
###################

print("doing MVI")

for i, data_name in enumerate(data_names):
    model_name = mvi_names[i]
    cluster_class_neworder, class_palette = make_palette_from_meta(data_name)
    column_names = ["UMAP D1", "UMAP D2"]
    if not os.path.exists(metric_dir + "mvi_" + data_name + "_TCELL_umap.csv"):
        print("making umap for data " + data_name)
        data = ad.read_h5ad(data_dir + "human_bonemarrow.h5ad")
        modality_switch = 13431
        train_indices = list(np.where(data.obs["train_val_test"] == "train")[0])
        trainset = data[train_indices, :]

        trainset.var_names_make_unique()
        trainset.obs["modality"] = "paired"
        if data_name == "human_bonemarrow":
            trainset.X = trainset.layers["counts"]
        if data_name != "human_brain":
            scvi.model.MULTIVI.setup_anndata(trainset, batch_key=mvi_batch_keys[i])
        else:
            scvi.model.MULTIVI.setup_anndata(trainset)
        model = scvi.model.MULTIVI.load(
            save_dir + "multiVI/" + data_name + "/" + model_name, adata=trainset
        )

        t_cell_string = "+ T"
        t_cell_indices = [
            i
            for i, s in enumerate(trainset.obs["cell_type"].values)
            if t_cell_string in s
        ]
        # np.where(np.char.find(trainset.obs['cell_type'].values, t_cell_string) != -1)[0]

        # get latent spaces in reduced dimensionality
        rep = model.get_latent_representation()[t_cell_indices, :]
        model = None
        cell_labels = trainset.obs[plotting_key[i]].values[t_cell_indices]

        # make umap
        n_neighbors = mvi_neighbors[i]
        min_dist = mvi_dists[i]
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist)
        projected = reducer.fit_transform(rep)
        plot_data = pd.DataFrame(projected, columns=column_names)
        plot_data["cell type"] = cell_labels
        plot_data["cell type"] = plot_data["cell type"].astype("category")
        # plot_data['cell type'] = plot_data['cell type'].cat.set_categories(cluster_class_neworder)
        # plot_data['data set'] = 'train'
        plot_data.to_csv(
            metric_dir + "mvi_" + data_name + "_TCELL_umap.csv", index=False
        )
    else:
        plot_data = pd.read_csv(metric_dir + "mvi_" + data_name + "_TCELL_umap.csv")
        plot_data["cell type"] = plot_data["cell type"].astype("category")
        plot_data["cell type"] = plot_data["cell type"].cat.set_categories(
            cluster_class_neworder
        )

    ax_list.append(plt.subplot(gs[1]))
    ax_list[-1].text(
        grid_letter_positions[0],
        1.0 + grid_letter_positions[1],
        "B",
        transform=ax_list[-1].transAxes + trans,
        fontsize=grid_letter_fontsize,
        va="bottom",
        fontfamily=grid_letter_fontfamily,
        fontweight=grid_letter_fontweight,
    )
    sns.scatterplot(
        data=plot_data,  # .sort_values(by='cell type'),
        x=column_names[0],
        y=column_names[1],
        hue="cell type",
        palette=class_palette,
        ax=ax_list[-1],
        s=point_size,
        alpha=alpha,
        linewidth=point_linewidth,
    )
    # legend
    ax_list[-1].legend(
        bbox_to_anchor=(1.0 + legend_x_dist, 1.0 + legend_y_dist),
        loc="upper left",
        frameon=False,
        handletextpad=handletextpad,
        markerscale=handlesize,
        fontsize=3,
    ).remove()
    ax_list[-1].set_title("MultiVI")
    # remove axis ticks
    ax_list[-1].tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False
    )
    # also remove axis tick values
    ax_list[-1].set_xticklabels([])
    ax_list[-1].set_yticklabels([])


figure_name = (
    "../results/analysis/plots/supplementaries/fig_embeddings_supp_marrow_tcells"
)
plt.savefig(figure_name + ".png", dpi=720, bbox_inches="tight")
