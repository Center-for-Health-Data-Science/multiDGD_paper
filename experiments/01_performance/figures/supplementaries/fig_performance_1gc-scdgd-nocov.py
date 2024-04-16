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
figure_height = 16
n_cols = 2
n_rows = 14
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=0.4, hspace=10.0)
ax_list = []
palette_models = ["#015799", "#DAA327", "#BDE1CD", "palegoldenrod"]
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

random_seeds = [0, 37, 8790]*4
model_types = ["", "", "", "", "", "", "noCovariate", "noCovariate", "noCovariate", "scDGD", "scDGD", "scDGD"]
n_components = [22, 22, 22, 1, 1, 1, 22, 22, 22, 22, 22, 22]
model_descriptors = ["default", "default", "default", "single Gaussian", "single Gaussian", "single Gaussian", "no covariate", "no covariate", "no covariate", "scDGD", "scDGD", "scDGD"]
n_features_bm = 129921
modality_switch = 13431

data_name = "human_bonemarrow"
result_path = "../results/revision/analysis/batch_integration/" + data_name + "/"
plot_path = "../results/revision/plots/"

# load all prediction errors
prediction_errors = pd.DataFrame()
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
    #temp_df = temp_df[["rmse", "batch"]]
    temp_df["model"] = model_name
    temp_df["rs"] = random_seeds[i]
    prediction_errors = pd.concat([prediction_errors, temp_df])
prediction_errors["normalized error"] = prediction_errors["rmse"] / modality_switch
# set the order of the models
prediction_errors["model"] = pd.Categorical(
    prediction_errors["model"],
    categories=["default", "single Gaussian", "no covariate", "scDGD"],
    ordered=True,
)

clustering_df = pd.read_csv(os.path.join(result_path, "human_bonemarrow_clustering_metrics_2.csv"))

####################
# plot
####################

ax_list.append(fig.add_subplot(gs[0:3, 0]))
ax_list[-1].text(
    grid_letter_positions[0],
    1.0 + 2 * grid_letter_positions[1],
    "A",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
sns.pointplot(
    x="model",
    y="rmse",
    data=prediction_errors,
    hue="model",
    palette=palette_models,
    ax=ax_list[-1],
    join=False,
    #dodge=0.5,
    markers=".",
    scale=0.5,
    errorbar="se",
    errwidth=0.5,
    capsize=0.2
)
labels = ax_list[-1].get_xticklabels()
new_labels = []
for label in labels:
    new_labels.append(label.get_text().replace(" ", "\n"))
ax_list[-1].set_xticklabels(new_labels)
ax_list[-1].set_ylabel("RMSE (sample-wise)")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("Reconstruction (RNA) \u2191")
ax_list[-1].legend().remove()

ax_list.append(fig.add_subplot(gs[0:3, 1]))
sns.pointplot(
    x="model",
    y="auprc",
    data=prediction_errors[prediction_errors["model"] != "scDGD"],
    hue="model",
    palette=palette_models[:3],
    ax=ax_list[-1],
    join=False,
    #dodge=0.5,
    markers=".",
    scale=0.5,
    errorbar="se",
    errwidth=0.5,
    capsize=0.2
)
ax_list[-1].set_xticklabels(new_labels)
ax_list[-1].set_ylabel("AUPRC (sample-wise)")
ax_list[-1].set_xlabel("Model")
# title
ax_list[-1].set_title("Reconstruction (ATAC) \u2191")
ax_list[-1].legend().remove()

ax_list.append(fig.add_subplot(gs[4:7, 0]))
ax_list[-1].text(
    grid_letter_positions[0],
    1.0 + 2 * grid_letter_positions[1],
    "B",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
sns.pointplot(
    x="model",
    y="ARI (Leiden)",
    data=clustering_df,
    hue="model",
    palette=palette_models,
    ax=ax_list[-1],
    join=False,
    #dodge=0.5,
    markers=".",
    scale=0.5,
    errorbar="se",
    errwidth=0.5,
    capsize=0.2
)
ax_list[-1].set_xticklabels(new_labels)
ax_list[-1].set_ylabel("ARI (Leiden)")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("Clustering \u2191")
ax_list[-1].legend(
    title="model",
    bbox_to_anchor=(0.5, -0.4),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad * 2,
    ncol=4,
)

clustering_df["1 - ASW"] = 1 - clustering_df["silhouette"]
ax_list.append(fig.add_subplot(gs[4:7, 1]))
sns.pointplot(
    x="model",
    y="1 - ASW",
    data=clustering_df,
    hue="model",
    palette=palette_models,
    ax=ax_list[-1],
    join=False,
    #dodge=0.5,
    markers=".",
    scale=0.5,
    errorbar="se",
    errwidth=0.5,
    capsize=0.2
)
ax_list[-1].set_xticklabels(new_labels)
#ax_list[-1].set_ylabel("ASW")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("Batch effect removal \u2191")
ax_list[-1].legend().remove()

# show two umaps

from omicsdgd import DGD

model_names = [
    "human_bonemarrow_l20_h2-3_test50e",
    "human_bonemarrow_l20_h2-3_rs37",
    "human_bonemarrow_l20_h2-3_rs8790",
    "human_bonemarrow_l20_h2-3_rs0_ncomp1_test50e",
    "human_bonemarrow_l20_h2-3_rs37_ncomp1_test50e",
    "human_bonemarrow_l20_h2-3_rs8790_ncomp1_test50e",
    "human_bonemarrow_l20_h2-3_rs0_noCovariate_test50e",
    "human_bonemarrow_l20_h2-3_rs37_noCovariate_test50e",
    "human_bonemarrow_l20_h2-3_rs8790_noCovariate_test50e",
    "human_bonemarrow_l20_h3_rs0_scDGD_test50e",
    "human_bonemarrow_l20_h3_rs37_scDGD_test50e",
    "human_bonemarrow_l20_h3_rs8790_scDGD_test50e"
]

# load umap of the representations
if not os.path.exists(os.path.join(result_path, "human_bonemarrow_umap_representations_scDGD_rs0.csv")):
    import anndata as ad
    # load data
    save_dir = "../results/trained_models/"
    data_name = "human_bonemarrow"
    adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
    adata.X = adata.layers["counts"]
    train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
    test_indices = list(np.where(adata.obs["train_val_test"] == "test")[0])
    trainset = adata[train_indices, :].copy()
    adata = None

    model_name = "human_bonemarrow_l20_h3_rs0_scDGD_test50e"
    modality_switch = 13431
    trainset = trainset[:, :modality_switch]
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
    rep_df.to_csv(os.path.join(result_path, "human_bonemarrow_umap_representations_scDGD_rs0.csv"), index=False)
    # save gmm
    np.savetxt(os.path.join(result_path, "human_bonemarrow_gmm_means_scDGD_rs0.csv"), gmm)
else:
    rep_df = pd.read_csv(os.path.join(result_path, "human_bonemarrow_umap_representations_scDGD_rs0.csv"))
    gmm = np.loadtxt(os.path.join(result_path, "human_bonemarrow_gmm_means_scDGD_rs0.csv"))
    
from omicsdgd.functions._analysis import make_palette_from_meta
batch_palette = ["#EEE7A8", "cornflowerblue", "darkmagenta", "darkslategray"]
data_name = "human_bonemarrow"
cluster_class_neworder, class_palette = make_palette_from_meta(data_name)

ax_list.append(fig.add_subplot(gs[9:, 0]))
ax_list[-1].text(
    grid_letter_positions[0],
    1.0 + 2 * grid_letter_positions[1],
    "C",
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
ax_list[-1].set_title("scDGD basal representation")
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
ax_list[-1].legend(
    title="cell type",
    bbox_to_anchor=(-0.2, -0.3),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad * 2,
    markerscale=handlesize,
    ncol=3,
)
# same plot by covariate
ax_list.append(fig.add_subplot(gs[9:, 1]))
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
ax_list[-1].set_title("scDGD basal representation")
ax_list[-1].legend(
    title="site",
    bbox_to_anchor=(0.02, -0.3),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad * 2,
    markerscale=handlesize,
    ncol=4,
)

# save
fig.savefig(os.path.join(plot_path, "performance_human_bonemarrow_model_setups.png"), bbox_inches="tight", dpi=300)