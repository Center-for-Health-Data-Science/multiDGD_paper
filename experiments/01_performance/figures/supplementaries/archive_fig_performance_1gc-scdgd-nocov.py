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
figure_height = 12
n_cols = 2
n_rows = 8
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=0.3, hspace=2.0)
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

def standard_error(data):
    return np.std(data) / np.sqrt(len(data))

from scipy.stats import ttest_ind

rmse_t_names = []
rmse_t_statistics = []
rmse_t_pvalues = []
auprc_t_names = []
auprc_t_statistics = []
auprc_t_pvalues = []
def get_significance(df, column, mod):
    t, p = ttest_ind(
        df[df["model"] == "default"][column],
        df[df["model"] == mod][column],
    )
    return t, p

def plot_significances(ax, df, column, offset, better="lower"):
    significances = []
    colors = []
    for mod in prediction_errors["model"].unique():
        if mod == "default":
            significances.append("")
            colors.append("black")
            continue
        t, p = ttest_ind(
            df[df["model"] == "default"][column],
            df[df["model"] == mod][column],
        )
        if p < 0.05:
            significances.append("*")
        else:
            significances.append("")
        if better == "lower":
            if df[df["model"] == "default"][column].mean() > df[df["model"] == mod][column].mean():
                colors.append("black")
            else:
                colors.append("red")
        else:
            if df[df["model"] == "default"][column].mean() < df[df["model"] == mod][column].mean():
                colors.append("black")
            else:
                colors.append("red")
    # plot the significance on the plot
    for i, mod in enumerate(df["model"].unique()):
        # get the heigth of the bar
        height = df[df["model"] == mod][column].mean() #+ prediction_errors_recon[prediction_errors_recon["model"] == model][column].std()
        # add the standard error of the mean
        height = height + standard_error(df[df["model"] == mod][column])
        height = height + offset
        ax.text(
            i,
            height,
            significances[i],
            ha="center",
            va="center",
            color=colors[i],
            fontsize=6,
            fontweight="bold",
            path_effects=[PathEffects.withStroke(linewidth=0.5, foreground="white")],
        )

####################
# get data
####################

# compare performance between normal multiDGD, multiDGD with a single Gaussian, multiDGD without covariate modeling, and scDGD

random_seeds = [0, 37, 8790]*6
model_types = ["", "", "", "l40", "l40", "l40", "", "", "", "noCovariate", "noCovariate", "noCovariate", "scDGD", "scDGD", "scDGD", "scDGD-ATAC", "scDGD-ATAC", "scDGD-ATAC"]
n_components = [22, 22, 22, 22, 22, 22, 1, 1, 1, 22, 22, 22, 22, 22, 22, 22, 22, 22]
model_descriptors = ["default", "default", "default", "double latent", "double latent", "double latent", "single Gaussian", "single Gaussian", "single Gaussian", "no covariate", "no covariate", "no covariate", "scDGD", "scDGD", "scDGD", "scDGD (ATAC)", "scDGD (ATAC)", "scDGD (ATAC)"]
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
    #temp_df["model_type"] = model_types[i]
    temp_df["rs"] = random_seeds[i]
    prediction_errors = pd.concat([prediction_errors, temp_df])
prediction_errors["normalized error"] = prediction_errors["rmse"] / modality_switch
# set the order of the models
prediction_errors["model"] = pd.Categorical(
    prediction_errors["model"],
    categories=["default", "double latent", "single Gaussian", "no covariate", "scDGD", "scDGD (ATAC)"],
    ordered=True,
)

#clustering_df = pd.read_csv(os.path.join(result_path, "human_bonemarrow_clustering_metrics_2.csv"))
clustering_df = pd.read_csv(os.path.join(result_path, "human_bonemarrow_clustering_metrics_3.csv"))

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
prediction_errors_recon = prediction_errors[prediction_errors["model"] != "scDGD (ATAC)"]
prediction_errors_recon["model"] = pd.Categorical(
    prediction_errors_recon["model"],
    categories=["default", "double latent", "single Gaussian", "no covariate", "scDGD"],
    ordered=True,
)
# compute significance of the difference between the default model and the other models
plot_significances(ax_list[-1], prediction_errors_recon, "rmse", 0.01, better="lower")
print("Number of samples: ", len(prediction_errors_recon[prediction_errors_recon["model"] == "default"]["rmse"]))
for mod in ["double latent", "single Gaussian", "no covariate", "scDGD"]:
    rmse_t_names.append(mod)
    t_test_temp = get_significance(prediction_errors_recon, "rmse", mod)
    rmse_t_statistics.append(t_test_temp[0])
    rmse_t_pvalues.append(t_test_temp[1])
#ax_list[-1].set_ylim(0.48, 0.8)
# plot a grey line on the mean of the default model
mean_default = prediction_errors_recon[prediction_errors_recon["model"] == "default"]["rmse"].mean()
ax_list[-1].axhline(mean_default, color="grey", linestyle="--", linewidth=0.3)
sns.pointplot(
    x="model",
    y="rmse",
    #data=prediction_errors,
    data=prediction_errors_recon,
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
#ax_list[-1].set_title("Reconstruction (RNA) \u2191")
# use arrow down
ax_list[-1].set_title("Test reconstruction error (RNA) \u2193")
ax_list[-1].legend().remove()

ax_list.append(fig.add_subplot(gs[0:3, 1]))
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
prediction_errors_recon = prediction_errors[prediction_errors["model"] != "scDGD"]
prediction_errors_recon["model"] = pd.Categorical(
    prediction_errors_recon["model"],
    categories=["default", "double latent", "single Gaussian", "no covariate", "scDGD (ATAC)"],
    ordered=True,
)
plot_significances(ax_list[-1], prediction_errors_recon, "auprc", 0.0002, better="higher")
for mod in ["double latent", "single Gaussian", "no covariate", "scDGD (ATAC)"]:
    auprc_t_names.append(mod)
    t_test_temp = get_significance(prediction_errors_recon, "auprc", mod)
    auprc_t_statistics.append(t_test_temp[0])
    auprc_t_pvalues.append(t_test_temp[1])
df_test_statistics = pd.DataFrame({
    "model": rmse_t_names,
    "statistic": rmse_t_statistics,
    "pvalue": rmse_t_pvalues,
    "type": "rmse"
})
df_test_statistics = pd.concat([
    df_test_statistics,
    pd.DataFrame({
        "model": auprc_t_names,
        "statistic": auprc_t_statistics,
        "pvalue": auprc_t_pvalues,
        "type": "auprc"
    })
])
# save 
df_test_statistics.to_csv(os.path.join("../results/revision/analysis/performance/human_bonemarrow_model_setups_statistics.csv"), index=False)

#ax_list[-1].set_ylim(0.2, 0.23)
mean_default = prediction_errors_recon[prediction_errors_recon["model"] == "default"]["auprc"].mean()
ax_list[-1].axhline(mean_default, color="grey", linestyle="--", linewidth=0.3)
sns.pointplot(
    x="model",
    y="auprc",
    #data=prediction_errors[prediction_errors["model"] != "scDGD"],
    data=prediction_errors_recon,
    hue="model",
    palette=palette_models[:4] + palette_models[-1:],
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
ax_list[-1].set_ylabel("AUPRC (sample-wise)")
ax_list[-1].set_xlabel("Model")
# title
ax_list[-1].set_title("Test reconstruction (ATAC) \u2191")
ax_list[-1].legend().remove()

ax_list.append(fig.add_subplot(gs[4:7, 0]))
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
#plot_significances(ax_list[-1], clustering_df, "ARI (Leiden)", 0.01, better="higher")
#ax_list[-1].set_ylim(0.36, 0.68)
mean_default = clustering_df[clustering_df["model"] == "default"]["ARI (Leiden)"].mean()
print("Number of samples in clustering data: ", len(clustering_df[clustering_df["model"] == "default"]["ARI (Leiden)"]))
ax_list[-1].axhline(mean_default, color="grey", linestyle="--", linewidth=0.3)
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
sns.stripplot(
    x="model",
    y="ARI (Leiden)",
    data=clustering_df,
    color="black",
    ax=ax_list[-1],
    size=1.5,
)
labels = ax_list[-1].get_xticklabels()
new_labels = []
for label in labels:
    new_labels.append(label.get_text().replace(" ", "\n"))
ax_list[-1].set_xticklabels(new_labels)
ax_list[-1].set_ylabel("ARI (Leiden)")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("Clustering \u2191")
ax_list[-1].legend(
    title="model",
    bbox_to_anchor=(0.1, -0.5),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad * 2,
    ncol=6,
)

clustering_df["1 - ASW"] = 1 - clustering_df["silhouette"]
ax_list.append(fig.add_subplot(gs[4:7, 1]))
ax_list[-1].text(
    grid_letter_positions[0],
    1.0 + 2 * grid_letter_positions[1],
    "D",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
#plot_significances(ax_list[-1], clustering_df, "1 - ASW", 0.002, better="higher")
#ax_list[-1].set_ylim(1.015, 1.075)
mean_default = clustering_df[clustering_df["model"] == "default"]["1 - ASW"].mean()
ax_list[-1].axhline(mean_default, color="grey", linestyle="--", linewidth=0.3)
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
sns.stripplot(
    x="model",
    y="1 - ASW",
    data=clustering_df,
    color="black",
    ax=ax_list[-1],
    size=1.5,
)
ax_list[-1].set_xticklabels(new_labels)
#ax_list[-1].set_ylabel("ASW")
ax_list[-1].set_xlabel("Model")
ax_list[-1].set_title("Batch effect removal \u2191")
ax_list[-1].legend().remove()

# save
fig.savefig(os.path.join(plot_path, "performance_human_bonemarrow_model_setups_main.png"), bbox_inches="tight", dpi=300)