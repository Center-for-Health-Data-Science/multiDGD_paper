import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
import anndata as ad
import seaborn as sns
import pandas as pd
import numpy as np
import os

####################################
# flexible parameters
####################################
figure_height = 8
n_cols = 3
n_rows = 2
grid_wspace = 0.5
grid_hspace = 0.5
figure_name = "../results/analysis/plots/supplementaries/fig_batch_supp"
analysis_dir = "../results/analysis/"

####################################
# fixed figure design
####################################
# set up figure and grid
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=grid_wspace, hspace=grid_hspace)
ax_list = []
# fonts
# general text
# text_size = 8
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
legend_x_dist, legend_y_dist = -0.02, 0.0
handletextpad = 0.1
# scatter plot
point_size = 0.3
alpha = 1
hist_alpha = 0.3
point_linewidth = 0.0
handlesize = 0.5
# line plot
linewidth = 0.5

# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)

####################################
# figure template
####################################

# suplots A and B: plot the distributions of counts colored by the site (A for RNA, B for ATAC)
# load the data set
save_dir = "results/trained_models/"
data_name = "human_bonemarrow"

if not os.path.exists(analysis_dir + "supplementaries/"):
   os.makedirs(analysis_dir + "supplementaries/")

# """
count_type = "sum"
if not os.path.exists(analysis_dir + "supplementaries/fig_batch_supp_samples_rna.csv"):
    #trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
    import anndata as ad

    data_name = "human_bonemarrow"
    adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
    adata.X = adata.layers["counts"]
    train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
    test_indices = list(np.where(adata.obs["train_val_test"] == "test")[0])
    trainset = adata[train_indices, :].copy()
    testset = adata[test_indices, :].copy()
    modality_switch = 13431

    print("loaded")
    print("n_features_atac: ", trainset.X.shape[1] - modality_switch)
    if count_type == "sum":
        x_rna = np.asarray(trainset.X[:, :modality_switch].sum(axis=1)).flatten()
        df_temp = pd.DataFrame(x_rna, columns=["counts"])
        df_temp["n_zeros"] = np.asarray(
            (trainset.X[:, :modality_switch] == 0).sum(axis=1)
        ).flatten()
        # turn n_zeros into fraction
        df_temp["n_zeros"] = df_temp["n_zeros"] / modality_switch
        df_temp["batch"] = trainset.obs["Site"].values
        data_rna = df_temp
    else:
        subset_idx = np.random.choice(trainset.X.shape[0], 1000, replace=False)
        sites = trainset.obs["Site"].values[subset_idx]
        for i in range(len(np.unique(sites))):
            site = np.unique(sites)[i]
            subset_subset_idx = np.where(sites == site)[0]
            x_rna_site = np.asarray(
                trainset.X[subset_idx[subset_subset_idx], :modality_switch].todense()
            ).flatten()
            df_temp = pd.DataFrame(x_rna_site, columns=["counts"])
            df_temp["batch"] = site
            if i == 0:
                data_rna = df_temp
            else:
                data_rna = pd.concat([data_rna, df_temp], axis=0)
    data_rna["modality"] = "RNA"
    trainset, testset, library = None, None, None
    # save the data
    data_rna.to_csv(
        analysis_dir+"supplementaries/fig_batch_supp_samples_rna.csv", index=False
    )
    print("done")

    print("atac")
    #trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
    import anndata as ad

    data_name = "human_bonemarrow"
    adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
    adata.X = adata.layers["counts"]
    train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
    test_indices = list(np.where(adata.obs["train_val_test"] == "test")[0])
    trainset = adata[train_indices, :].copy()
    testset = adata[test_indices, :].copy()
    modality_switch = 13431

    print("loaded")
    if count_type == "sum":
        # get the counts for the whole data in batches of 1000
        for i in range(0, trainset.X.shape[0], 1000):
            subset_idx = np.arange(i, min(i + 1000, trainset.X.shape[0]))
            # subset_idx = np.random.choice(trainset.X.shape[0], 1000, replace=False)
            sites = trainset.obs["Site"].values[subset_idx]
            x_atac = np.asarray(
                trainset.X[subset_idx, modality_switch:].sum(axis=1)
            ).flatten()
            df_temp = pd.DataFrame(x_atac, columns=["counts"])
            x_atac = None
            # get zero counts from sparse matrix
            df_temp["n_zeros"] = np.asarray(
                (trainset.X[subset_idx, modality_switch:] == 0).sum(axis=1)
            ).flatten()
            df_temp["batch"] = sites
            if i == 0:
                data_atac = df_temp
            else:
                data_atac = pd.concat([data_atac, df_temp], axis=0)
    else:
        subset_idx = np.random.choice(trainset.X.shape[0], 1000, replace=False)
        sites = trainset.obs["Site"].values[subset_idx]
        for i in range(len(np.unique(sites))):
            site = np.unique(sites)[i]
            subset_subset_idx = np.where(sites == site)[0]
            x_atac_site = np.asarray(
                trainset.X[subset_idx[subset_subset_idx], modality_switch:].todense()
            ).flatten()
            df_temp = pd.DataFrame(x_atac_site, columns=["counts"])
            df_temp["batch"] = site
            if i == 0:
                data_atac = df_temp
            else:
                data_atac = pd.concat([data_atac, df_temp], axis=0)
    data_atac["modality"] = "ATAC"
    trainset, testset, library = None, None, None
    # save the data
    data_atac.to_csv(
        analysis_dir+"supplementaries/fig_batch_supp_samples_atac.csv", index=False
    )
    print("done")

else:
    print("already created")
    data_rna = pd.read_csv(
        analysis_dir+"supplementaries/fig_batch_supp_samples_rna.csv"
    )
    data_atac = pd.read_csv(
        analysis_dir+"supplementaries/fig_batch_supp_samples_atac.csv"
    )

data_atac["n_zeros"] = data_atac["n_zeros"] / 116490
data = pd.concat([data_rna, data_atac], axis=0)
data["batch"] = data["batch"].astype("category")
data["batch"] = data["batch"].cat.set_categories(["site1", "site2", "site3", "site4"])
# print(data)

# first subplot
ax_list.append(plt.subplot(gs[0, 0]))
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
# set title
ax_list[-1].set_title("Count depths")
###
# here comes the plotting
###
sns.histplot(
    data=data[data["modality"] == "RNA"],
    x="counts",
    hue="batch",
    bins=100,
    palette=batch_palette,
    ax=ax_list[-1],
    alpha=hist_alpha,
    element="step",
)
# print legend handles
handles, labels = ax_list[-1].get_legend_handles_labels()
# print('handles', handles)
# print('labels', labels)
# ax_list[-1].legend(bbox_to_anchor=(1.+legend_x_dist, 1.+legend_y_dist),
#                   loc='upper left', frameon=False,
#                   handletextpad=handletextpad)
# print legend handles
# handles, labels = ax_list[-1].get_legend_handles_labels()
# print('handles', handles)
# print('labels', labels)
ax_list[-1].get_legend().remove()
ax_list[-1].set_xlabel("RNA count depth")
ax_list[-1].set_ylabel("Frequency")
ax_list[-1].set_xlim([0, 10000])
# ax_list[-1].set_xscale('log')
# ax_list[-1].set_yscale('log')

# second subplot
ax_list.append(plt.subplot(gs[1, 0]))
sns.histplot(
    data=data[data["modality"] == "ATAC"],
    x="counts",
    hue="batch",
    bins=100,
    palette=batch_palette,
    ax=ax_list[-1],
    alpha=hist_alpha,
    element="step",
)
# ax_list[-1].legend(bbox_to_anchor=(1.+legend_x_dist, 1.+legend_y_dist),
#                  loc='upper left', frameon=False,
#                  handletextpad=handletextpad)
ax_list[-1].get_legend().remove()
ax_list[-1].set_xlabel("ATAC count depth")
ax_list[-1].set_ylabel("Frequency")
ax_list[-1].set_xlim([0, 40000])
# ax_list[-1].set_xscale('log')
# ax_list[-1].set_yscale('log')
# """

# new row (number of zeros)
ax_list.append(plt.subplot(gs[0, 1]))
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
# set title
ax_list[-1].set_title("Fraction of zero counts")
###
# here comes the plotting
###
sns.histplot(
    data=data[data["modality"] == "RNA"],
    x="n_zeros",
    hue="batch",
    bins=100,
    palette=batch_palette,
    ax=ax_list[-1],
    alpha=hist_alpha,
    element="step",
)
# ax_list[-1].legend(bbox_to_anchor=(1.+legend_x_dist, 1.+legend_y_dist),
#                   loc='upper left', frameon=False,
#                   handletextpad=handletextpad)
ax_list[-1].get_legend().remove()
ax_list[-1].set_xlabel("RNA fraction of zero counts")
ax_list[-1].set_ylabel("Frequency")
# ax_list[-1].set_xscale('log')
# ax_list[-1].set_yscale('log')

# second subplot
ax_list.append(plt.subplot(gs[1, 1]))
sns.histplot(
    data=data[data["modality"] == "ATAC"],
    x="n_zeros",
    hue="batch",
    bins=100,
    palette=batch_palette,
    ax=ax_list[-1],
    alpha=hist_alpha,
    element="step",
)
# ax_list[-1].legend(bbox_to_anchor=(1.+legend_x_dist, 1.+legend_y_dist),
#                  loc='upper left', frameon=False,
#                  handletextpad=handletextpad)
# remove the legend
ax_list[-1].get_legend().remove()
ax_list[-1].set_xlabel("ATAC fraction of zero counts")
ax_list[-1].set_ylabel("Frequency")
# ax_list[-1].set_xscale('log')
# ax_list[-1].set_yscale('log')

# fifth subplot
# plot pca of the whole data colored by the site
if not os.path.exists(
    analysis_dir+"supplementaries/fig_batch_supp_samples_pca.csv"
):
    step = 10

    import anndata as ad

    data_name = "human_bonemarrow"
    adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
    adata.X = adata.layers["counts"]
    train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
    test_indices = list(np.where(adata.obs["train_val_test"] == "test")[0])
    trainset = adata[train_indices, :].copy()
    testset = adata[test_indices, :].copy()
    modality_switch = 13431

    #trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
    print("loaded")
    library = None
    sites = trainset.obs["Site"].values[::step]
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    count_depths_rna = np.asarray(
        trainset.X[::step, :modality_switch].sum(axis=1)
    )  # .flatten()
    # print(count_depths_rna.shape)
    count_depths_atac = np.asarray(
        trainset.X[::step, modality_switch:].sum(axis=1)
    )  # .flatten()
    # counts = trainset.X[::step,:].toarray()
    counts_rna = trainset.X[::step, :modality_switch].toarray()
    counts_rna /= count_depths_rna
    counts_atac = trainset.X[::step, modality_switch:].toarray()
    trainset, testset = None, None
    counts_atac /= count_depths_atac
    counts = np.concatenate([counts_rna, counts_atac], axis=1)
    counts_rna, counts_atac = None, None
    # counts[:, :modality_switch] = counts[:, :modality_switch] / count_depths_rna[:,None]
    # counts[:, modality_switch:] = counts[:, modality_switch:] / count_depths_atac[:,None]
    # make log norm
    counts = np.log1p(counts)
    pca_data = pca.fit_transform(counts)
    pca_data = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
    pca_data["batch"] = sites
    pca_data.to_csv(
        analysis_dir+"supplementaries/fig_batch_supp_samples_pca.csv", index=False
    )
else:
    pca_data = pd.read_csv(
        analysis_dir+"supplementaries/fig_batch_supp_samples_pca.csv"
    )
pca_data["batch"] = pca_data["batch"].astype("category")
pca_data["batch"] = pca_data["batch"].cat.set_categories(
    ["site1", "site2", "site3", "site4"]
)

ax_list.append(plt.subplot(gs[0, 2]))
ax_list[-1].text(
    grid_letter_positions[0],
    1.0 + grid_letter_positions[1],
    "C",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
# set title
ax_list[-1].set_title("PCA of normalized counts")
pca_data["batch"] = pca_data["batch"].astype("category")
pca_data["batch"] = pca_data["batch"].cat.set_categories(
    ["site1", "site2", "site3", "site4"]
)
sns.scatterplot(
    data=pca_data,
    x="PC1",
    y="PC2",
    hue="batch",
    palette=batch_palette,
    ax=ax_list[-1],
    s=point_size,
    alpha=alpha,
    linewidth=point_linewidth,
)
ax_list[-1].legend(
    bbox_to_anchor=(1.0 + legend_x_dist, 1.0 + legend_y_dist),
    loc="upper left",
    frameon=False,
    markerscale=handlesize,
    handletextpad=handletextpad,
)

# sixth subplot
# plot the batch representation
correction_df = pd.read_csv(
    analysis_dir+"performance_evaluation/bonemarrow_correction_umap.csv"
)
correction_df["batch"] = correction_df["batch"].astype("category")
correction_df["batch"] = correction_df["batch"].cat.set_categories(
    ["site1", "site2", "site3", "site4"]
)
ax_list.append(plt.subplot(gs[1, 2]))
ax_list[-1].text(
    grid_letter_positions[0],
    1.0 + grid_letter_positions[1],
    "D",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
# set title
ax_list[-1].set_title("Covariate representation")
sns.scatterplot(
    data=correction_df,
    x="D1",
    y="D2",
    hue="batch",
    palette=batch_palette,
    ax=ax_list[-1],
    s=point_size,
    alpha=alpha,
    linewidth=point_linewidth,
)
ax_list[-1].legend(
    bbox_to_anchor=(1.0 + legend_x_dist, 1.0 + legend_y_dist),
    loc="upper left",
    frameon=False,
    handletextpad=handletextpad,
    markerscale=handlesize,
)

# save figure
plt.savefig(figure_name + ".png", dpi=300, bbox_inches="tight")
