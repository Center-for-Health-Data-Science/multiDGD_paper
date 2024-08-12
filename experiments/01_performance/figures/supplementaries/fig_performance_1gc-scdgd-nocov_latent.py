import os
import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as PathEffects
import anndata as ad

version_1 = "multiDGD"
version_2 = "scDGD"

version = version_2

if version == version_1:
    model_names = [
        "human_bonemarrow_l20_h2-3_test50e",
        "human_bonemarrow_l40_h2-3_rs0_test50e",
        "human_bonemarrow_l20_h2-3_rs0_ncomp1_test50e",
        "human_bonemarrow_l20_h2-3_rs0_noCovariate_test50e"
    ]
    model_descriptors = ["default", "double latent", "single Gaussian", "no covariate"]

elif version == version_2:
    model_names = [
        "human_bonemarrow_l20_h2-3_test50e",
        "human_bonemarrow_l20_h3_rs0_scDGD_test50e",
        "human_bonemarrow_l20_h3_rs0_scDGD_atac_test50e"
    ]
    model_descriptors = ["default", "scDGD", "scDGD (ATAC)"]
else:
    raise ValueError("version not recognized")

# set up figure
n_rows = len(model_names)
figure_height = n_rows * 4
n_cols = 2
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=0.3, hspace=0.7)
ax_list = []
palette_models = ["#2B76B0", "#003760" , "#F4D35E", "#93E5AB", "#65B891", "#2A6A4E"]
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
alpha = 0.5
point_linewidth = 0.0
handlesize = 0.3

# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)

####################
# get data
####################

# compare performance between normal multiDGD, multiDGD with a single Gaussian, multiDGD without covariate modeling, and scDGD

#random_seeds = [0, 37, 8790]*6
#model_types = ["", "", "", "l40", "l40", "l40", "", "", "", "noCovariate", "noCovariate", "noCovariate", "scDGD", "scDGD", "scDGD", "scDGD-ATAC", "scDGD-ATAC", "scDGD-ATAC"]
#n_components = [22, 22, 22, 22, 22, 22, 1, 1, 1, 22, 22, 22, 22, 22, 22, 22, 22, 22]
#model_descriptors = ["default", "default", "default", "double latent", "double latent", "double latent", "single Gaussian", "single Gaussian", "single Gaussian", "no covariate", "no covariate", "no covariate", "scDGD", "scDGD", "scDGD", "scDGD (ATAC)", "scDGD (ATAC)", "scDGD (ATAC)"]
n_features_bm = 129921
modality_switch = 13431

data_name = "human_bonemarrow"
result_path = "../results/revision/analysis/batch_integration/" + data_name + "/"
plot_path = "../results/revision/plots/"

# show two umaps

from omicsdgd import DGD

alphabet_letters = ["A", "B", "C", "D", "E", "F"]

from omicsdgd.functions._analysis import make_palette_from_meta
batch_palette = ["#EEE7A8", "cornflowerblue", "darkmagenta", "darkslategray"]
data_name = "human_bonemarrow"
cluster_class_neworder, class_palette = make_palette_from_meta(data_name)

for i, model_name in enumerate(model_names):
    if not os.path.exists(os.path.join(result_path, model_name + "_umap_representations.csv")):
        print("making umap for ", model_name)
        # load data
        save_dir = "../results/trained_models/"
        adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
        adata.X = adata.layers["counts"]
        train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
        test_indices = list(np.where(adata.obs["train_val_test"] == "test")[0])

        if "scDGD_atac" in model_name:
            trainset = adata[train_indices, modality_switch:]
        elif "scDGD" in model_name:
            trainset = adata[train_indices, :modality_switch]
        else:
            trainset = adata[train_indices, :].copy()
        adata = None

        model = DGD.load(
            data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
        )
        # make a umap of the representations
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.5)
        rep = reducer.fit_transform(model.representation.z.detach().cpu().numpy())
        # get the gmm means
        gmm = reducer.transform(model.gmm.mean.detach().cpu().numpy())
        # put the data in a dataframe so I can randomize the order
        rep_df = pd.DataFrame(rep, columns=["UMAP D1", "UMAP D2"])
        rep_df["cell type"] = trainset.obs["cell_type"].values
        rep_df["Site"] = trainset.obs["Site"].values
        rep_df = rep_df.sample(frac=1)

        # save the umap
        rep_df.to_csv(os.path.join(result_path, model_name + "_umap_representations.csv"), index=False)
        # save gmm
        np.savetxt(os.path.join(result_path, model_name + "_gmm_means.csv"), gmm)
    else:
        print("loading umap for ", model_name)
        rep_df = pd.read_csv(os.path.join(result_path, model_name + "_umap_representations.csv"))
        gmm = np.loadtxt(os.path.join(result_path, model_name + "_gmm_means.csv"))

    ax_list.append(fig.add_subplot(gs[i, 0]))
    ax_list[-1].text(
        grid_letter_positions[0],
        1.0 + 2 * grid_letter_positions[1],
        alphabet_letters[i],
        transform=ax_list[-1].transAxes + trans,
        fontsize=grid_letter_fontsize,
        va="bottom",
        fontfamily=grid_letter_fontfamily,
        fontweight=grid_letter_fontweight,
    )
    sns.scatterplot(
        x="UMAP D1",
        y="UMAP D2",
        data=rep_df,
        hue="cell type",
        palette=class_palette,
        s=1,
        alpha=alpha,
        linewidth=0,
        ax=ax_list[-1]
    )
    # plot the gmm means
    ax_list[-1].set_title("'" + model_descriptors[i] + "'" + " basal representation")
    if "ncomp1" not in model_name:
        for j in range(gmm.shape[0]):
            ax_list[-1].text(
                gmm[j, 0], gmm[j, 1], 
                str(j), 
                fontsize=8,
                color="black",
                ha="center",
                va="center",
                fontweight="bold",
                path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="w")]
            )
    else:
        ax_list[-1].text(
            gmm[0], gmm[1], 
            "0", 
            fontsize=8,
            color="black",
            ha="center",
            va="center",
            fontweight="bold",
            path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="w")]
        )
    if i == len(model_names) - 1:
        ax_list[-1].legend(
            title="cell type",
            bbox_to_anchor=(-0.17, -0.5),
            loc=2,
            borderaxespad=0.0,
            frameon=False,
            handletextpad=handletextpad * 2,
            markerscale=handlesize,
            ncol=3,
        )
    else:
        ax_list[-1].legend().remove()
    # same plot by covariate
    ax_list.append(fig.add_subplot(gs[i, 1]))
    sns.scatterplot(
        x="UMAP D1",
        y="UMAP D2",
        data=rep_df,
        hue="Site",
        palette=batch_palette,
        s=1,
        alpha=alpha,
        linewidth=0,
        ax=ax_list[-1]
    )
    # make the model descriptor in the title bold
    ax_list[-1].set_title("'" + model_descriptors[i] + "'" + " batch representation")
    if i == len(model_names) - 1:
        ax_list[-1].legend(
            title="site",
            bbox_to_anchor=(0.05, -0.5),
            loc=2,
            borderaxespad=0.0,
            frameon=False,
            handletextpad=handletextpad * 2,
            markerscale=handlesize,
            ncol=4,
        )
    else:
        ax_list[-1].legend().remove()

# save
if version == version_1:
    fig.savefig(os.path.join(plot_path, "human_bonemarrow_model_setups_latents_multiDGD.png"), bbox_inches="tight", dpi=300)
elif version == version_2:
    fig.savefig(os.path.join(plot_path, "human_bonemarrow_model_setups_latents_scDGD.png"), bbox_inches="tight", dpi=300)