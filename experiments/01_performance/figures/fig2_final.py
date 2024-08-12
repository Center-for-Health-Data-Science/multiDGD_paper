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

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

save_dir = "../results/trained_models/"
metric_dir = "../results/analysis/performance_evaluation/"
data_dir = "../../data/"
data_name = "human_bonemarrow"
model_name = "human_bonemarrow_l20_h2-3_test10e"  # this is just a speficiation of the model after test set inference

# set up figure
figure_height = 14
n_cols = 8
n_rows = 8
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
#gs_bottom = gridspec.GridSpec(4, n_cols)
gs.update(wspace=6.0, hspace=10.0)
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
legend_x_dist, legend_y_dist = -0.0, 0.0
grid_letter_positions = [-0.1, 0.05]
grid_letter_fontsize = 7
grid_letter_fontfamily = "sans-serif"
grid_letter_fontweight = "bold"
point_size = 0.2
linewidth = 0.5
alpha = 1
point_linewidth = 0.0
handlesize = 0.3
strip_color = "black"
strip_size = 1.5
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
# add text underneath the legend
ax_list[-1].text(4.25, 0.08, "Metrics")
ax_list[-1].text(4.4, 0.04, "\u2191: Higher is better")
ax_list[-1].text(4.4, 0.00, "\u2193: Lower is better")

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
ax2.set_ylabel('Test loss ratio')
ax_list[-1].set_ylabel('Test loss ratio')


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
# perform statistical test to get significance
from scipy.stats import ttest_ind, mannwhitneyu
ttest_rmse_in5 = ttest_ind(
    mouse_df[(mouse_df['Model']=='MultiVI') & (mouse_df['Feature selection']=='in 5%')]['rmse'].values,
    mouse_df[(mouse_df['Model']=='multiDGD') & (mouse_df['Feature selection']=='in 5%')]['rmse'].values
)
ttest_rmse_all = ttest_ind(
    mouse_df[(mouse_df['Model']=='MultiVI') & (mouse_df['Feature selection']=='all')]['rmse'].values,
    mouse_df[(mouse_df['Model']=='multiDGD') & (mouse_df['Feature selection']=='all')]['rmse'].values
)
mwu_rmse_in5 = mannwhitneyu(
    mouse_df[(mouse_df['Model']=='MultiVI') & (mouse_df['Feature selection']=='in 5%')]['rmse'].values,
    mouse_df[(mouse_df['Model']=='multiDGD') & (mouse_df['Feature selection']=='in 5%')]['rmse'].values
)
mwu_rmse_all = mannwhitneyu(
    mouse_df[(mouse_df['Model']=='MultiVI') & (mouse_df['Feature selection']=='all')]['rmse'].values,
    mouse_df[(mouse_df['Model']=='multiDGD') & (mouse_df['Feature selection']=='all')]['rmse'].values
)
print(f"t-test RMSE in 5%: {ttest_rmse_in5}")
print(f"t-test RMSE all: {ttest_rmse_all}")
ax_list[-1].plot(
    [-0.1, 0.1],
    [2.42, 2.42],
    color='black',
    linewidth=0.5
)
ax_list[-1].text(
    0.0,
    2.45,
    "***",
    horizontalalignment='center',
    verticalalignment='center'
)
ax_list[-1].plot(
    [0.9, 1.1],
    [2.55, 2.55],
    color='black',
    linewidth=0.5
)
ax_list[-1].text(
    1.0,
    2.58,
    "***",
    horizontalalignment='center',
    verticalalignment='center'
)
ax_list[-1].set_ylim((1.6, 2.65))
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

sns.pointplot(data=mouse_df, x='Feature selection', y='auprc', hue='Model', ax=ax_list[-1],
              palette=palette_2colrs, markers=".", scale=pointplot_scale,
              errwidth=pointplot_errwidth, capsize=pointplot_capsize,
              linestyles=["", ""],errorbar="se", dodge=0.3
              )
#print(len(mouse_df.index.unique()))
print(f"number of cells in mouse test: {len(mouse_df[(mouse_df['Model']=='MultiVI') & (mouse_df['Feature selection']=='in 5%')]['auprc'].values)}")
ttest_auprc_in5 = ttest_ind(
    mouse_df[(mouse_df['Model']=='MultiVI') & (mouse_df['Feature selection']=='in 5%')]['auprc'].values,
    mouse_df[(mouse_df['Model']=='multiDGD') & (mouse_df['Feature selection']=='in 5%')]['auprc'].values
)
ttest_auprc_all = ttest_ind(
    mouse_df[(mouse_df['Model']=='MultiVI') & (mouse_df['Feature selection']=='all')]['auprc'].values,
    mouse_df[(mouse_df['Model']=='multiDGD') & (mouse_df['Feature selection']=='all')]['auprc'].values
)
mwu_auprc_in5 = mannwhitneyu(
    mouse_df[(mouse_df['Model']=='MultiVI') & (mouse_df['Feature selection']=='in 5%')]['auprc'].values,
    mouse_df[(mouse_df['Model']=='multiDGD') & (mouse_df['Feature selection']=='in 5%')]['auprc'].values
)
mwu_auprc_all = mannwhitneyu(
    mouse_df[(mouse_df['Model']=='MultiVI') & (mouse_df['Feature selection']=='all')]['auprc'].values,
    mouse_df[(mouse_df['Model']=='multiDGD') & (mouse_df['Feature selection']=='all')]['auprc'].values
)
print(f"t-test AUPRC in 5%: {ttest_auprc_in5}")
print(f"t-test AUPRC all: {ttest_auprc_all}")
print(f"Mouse recon has {len(mouse_df[mouse_df['Model']=='multiDGD'])} values per category")
# export the data for the statistical test
df_stats = pd.DataFrame()
df_stats['Feature selection'] = ['in 5%', 'all']
df_stats['RMSE T statistic'] = ttest_rmse_in5.statistic, ttest_rmse_all.statistic
df_stats['RMSE T p value'] = ttest_rmse_in5.pvalue, ttest_rmse_all.pvalue
df_stats['RMSE T degrees of freedom'] = ttest_rmse_in5.df, ttest_rmse_all.df
df_stats['AUPRC T statistic'] = ttest_auprc_in5.statistic, ttest_auprc_all.statistic
df_stats['AUPRC T p value'] = ttest_auprc_in5.pvalue, ttest_auprc_all.pvalue
df_stats['AUPRC T degrees of freedom'] = ttest_auprc_in5.df, ttest_auprc_all.df
df_stats['RMSE MWU statistic'] = mwu_rmse_in5.statistic, mwu_rmse_all.statistic
df_stats['RMSE MWU p value'] = mwu_rmse_in5.pvalue, mwu_rmse_all.pvalue
df_stats['AUPRC MWU statistic'] = mwu_auprc_in5.statistic, mwu_auprc_all.statistic
df_stats['AUPRC MWU p value'] = mwu_auprc_in5.pvalue, mwu_auprc_all.pvalue
#df_stats['RMSE confidence 95'] = ttest_rmse_in5.confidence_interval(), ttest_rmse_all.confidence_interval()
#df_stats['AUPRC confidence 95'] = ttest_auprc_in5.confidence_interval(), ttest_auprc_all.confidence_interval()
print(df_stats)
# export the data for the statistical test
df_stats.to_csv('../results/revision/analysis/performance/feature_efficiency_ttest.csv', index=False)
# remove legend
ax_list[-1].plot(
    [0.9, 1.1],
    [0.394, 0.394],
    color='black',
    linewidth=0.5
)
ax_list[-1].text(
    1.0,
    0.3945,
    "*",
    horizontalalignment='center',
    verticalalignment='center'
)
ax_list[-1].set_ylim((0.381, 0.3955))
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
    "../results/figures/fig2_final.pdf",
    dpi=300,
    bbox_inches="tight",
    format="pdf",
)
