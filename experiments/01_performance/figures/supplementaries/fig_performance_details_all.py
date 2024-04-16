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
figure_height = 4
n_cols = 4
n_rows = 1
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=0.8, hspace=0.0)
ax_list = []
palette_2colrs = ["#015799", "#DAA327"]
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
alpha = 1
point_linewidth = 0.0
handlesize = 0.3

# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)

####################
# plot sample-wise
####################

# get data

random_seeds = [0, 37, 8790]*3
data_names = ["human_bonemarrow"]*3 + ["mouse_gastrulation"]*3 + ["human_brain"]*3
data_names_plot = ["marrow"]*3 + ["gastrulation"]*3 + ["brain"]*3
n_components = [22, 22, 22, 37, 37, 37, 16, 16, 16]
plot_path = "../results/revision/plots/"

prediction_metrics_sample = pd.DataFrame()
summary_stats_sample = pd.DataFrame()
for i, data_name in enumerate(data_names):
    result_path = "../results/revision/analysis/batch_integration/" + data_name + "/"
    
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "__ncomp"
        + str(n_components[i])
        + "_RMSE-BA_samplewise.csv"
    )
    # only keep columns rmse and batch
    temp_df["rs"] = random_seeds[i]
    #temp_df["count depth"] = mean_expression_cell
    temp_df["Model"] = "multiDGD"
    temp_df["Data"] = data_names_plot[i]
    prediction_metrics_sample = pd.concat([prediction_metrics_sample, temp_df])
    # also load mvi
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_RMSE-BA_samplewise_mvi.csv"
    )
    temp_df["rs"] = random_seeds[i]
    #temp_df["count depth"] = mean_expression_cell
    temp_df["Model"] = "MultiVI"
    temp_df["Data"] = data_names_plot[i]
    prediction_metrics_sample = pd.concat([prediction_metrics_sample, temp_df])
    n_samples = len(temp_df)

    prediction_metrics_sample_temp = prediction_metrics_sample[prediction_metrics_sample["rs"] == random_seeds[i]]
    prediction_metrics_sample_temp = prediction_metrics_sample_temp[prediction_metrics_sample_temp["Data"] == data_names_plot[i]]

    summary_stats_sample_temp = prediction_metrics_sample_temp.groupby(["Model"]).agg(
        mean_ba=("ba", "mean"),
        std_ba=("ba", "std"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_auprc=("auprc", "mean"),
        std_auprc=("auprc", "std"),
        mean_spearman=("spearman", "mean"),
        std_spearman=("spearman", "std"),
    )

    # make new columns for sem
    summary_stats_sample_temp["sem_ba"] = summary_stats_sample_temp["std_ba"] / np.sqrt(n_samples)
    summary_stats_sample_temp["sem_rmse"] = summary_stats_sample_temp["std_rmse"] / np.sqrt(n_samples)
    summary_stats_sample_temp["sem_auprc"] = summary_stats_sample_temp["std_auprc"] / np.sqrt(n_samples)
    summary_stats_sample_temp["sem_spearman"] = summary_stats_sample_temp["std_spearman"] / np.sqrt(n_samples)

    summary_stats_sample_temp["Data"] = data_names_plot[i]
    summary_stats_sample_temp["n_samples"] = n_samples

####################

# a figure with 3 subplots, showing the three different metrics for the ATAC test performance

# RMSE
ax_list.append(fig.add_subplot(gs[0, 0]))
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.0 + grid_letter_positions[1],
    "A",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
sns.pointplot(
    x="Data",
    y="rmse",
    hue="Model",
    data=prediction_metrics_sample,
    ax=ax_list[-1],
    join=False,
    dodge=0.3,
    markers=".",
    scale=0.5,
    errorbar="se",
    errwidth=0.5,
    capsize=0.2,
    palette=palette_2colrs,
)
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_ylabel("RMSE (sample-wise)")
ax_list[-1].set_title("RNA (per cell) \u2193")
# remove legend
ax_list[-1].get_legend().remove()

# next auprc
ax_list.append(fig.add_subplot(gs[0, 1]))
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.0 + grid_letter_positions[1],
    "B",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
sns.pointplot(
    x="Data",
    y="auprc",
    hue="Model",
    data=prediction_metrics_sample,
    ax=ax_list[-1],
    join=False,
    dodge=0.3,
    markers=".",
    scale=0.5,
    errorbar="se",
    errwidth=0.5,
    capsize=0.2,
    palette=palette_2colrs,
)
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_ylabel("AUPRC (sample-wise)")
ax_list[-1].set_title("ATAC (per cell) \u2191")
ax_list[-1].get_legend().remove()

####################

# now the same features-wise

prediction_metrics_sample = pd.DataFrame()
summary_stats_sample = pd.DataFrame()
for i, data_name in enumerate(data_names):
    result_path = "../results/revision/analysis/batch_integration/" + data_name + "/"
    
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "__ncomp"
        + str(n_components[i])
        + "_RMSE-BA_genewise.csv"
    )
    # only keep columns rmse and batch
    temp_df["rs"] = random_seeds[i]
    #temp_df["count depth"] = mean_expression_cell
    temp_df["Model"] = "multiDGD"
    temp_df["Data"] = data_names_plot[i]
    prediction_metrics_sample = pd.concat([prediction_metrics_sample, temp_df])
    # also load mvi
    temp_df = pd.read_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_RMSE-BA_genewise_mvi.csv"
    )
    temp_df["rs"] = random_seeds[i]
    #temp_df["count depth"] = mean_expression_cell
    temp_df["Model"] = "MultiVI"
    temp_df["Data"] = data_names_plot[i]
    prediction_metrics_sample = pd.concat([prediction_metrics_sample, temp_df])
    n_samples = len(temp_df)

    prediction_metrics_sample_temp = prediction_metrics_sample[prediction_metrics_sample["rs"] == random_seeds[i]]
    prediction_metrics_sample_temp = prediction_metrics_sample_temp[prediction_metrics_sample_temp["Data"] == data_names_plot[i]]

    summary_stats_sample_temp = prediction_metrics_sample_temp.groupby(["Model"]).agg(
        mean_ba=("ba", "mean"),
        std_ba=("ba", "std"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_auprc=("auprc", "mean"),
        std_auprc=("auprc", "std"),
        mean_spearman=("spearman", "mean"),
        std_spearman=("spearman", "std"),
    )

    # make new columns for sem
    summary_stats_sample_temp["sem_ba"] = summary_stats_sample_temp["std_ba"] / np.sqrt(n_samples)
    summary_stats_sample_temp["sem_rmse"] = summary_stats_sample_temp["std_rmse"] / np.sqrt(n_samples)
    summary_stats_sample_temp["sem_auprc"] = summary_stats_sample_temp["std_auprc"] / np.sqrt(n_samples)
    summary_stats_sample_temp["sem_spearman"] = summary_stats_sample_temp["std_spearman"] / np.sqrt(n_samples)

    summary_stats_sample_temp["Data"] = data_names_plot[i]
    summary_stats_sample_temp["n_samples"] = n_samples

# RMSE
ax_list.append(fig.add_subplot(gs[0, 2]))
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.0 + grid_letter_positions[1],
    "C",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
sns.pointplot(
    x="Data",
    y="rmse",
    hue="Model",
    data=prediction_metrics_sample,
    ax=ax_list[-1],
    join=False,
    dodge=0.3,
    markers=".",
    scale=0.5,
    errorbar="se",
    errwidth=0.5,
    capsize=0.2,
    palette=palette_2colrs,
)
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_ylabel("RMSE (gene-wise)")
ax_list[-1].set_title("RNA (per gene) \u2193")
ax_list[-1].get_legend().remove()

# next auprc
ax_list.append(fig.add_subplot(gs[0, 3]))
ax_list[-1].text(
    grid_letter_positions[0] * 2,
    1.0 + grid_letter_positions[1],
    "D",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
sns.pointplot(
    x="Data",
    y="auprc",
    hue="Model",
    data=prediction_metrics_sample,
    ax=ax_list[-1],
    dodge=0.3,
    join=False,
    markers=".",
    scale=0.5,
    errorbar="se",
    errwidth=0.5,
    capsize=0.2,
    palette=palette_2colrs,
)
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_ylabel("AUPRC (peak-wise)")
ax_list[-1].set_title("ATAC (per peak) \u2191")
ax_list[-1].legend(
    title="Model",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
    alignment="left",
    columnspacing=0.3,
    handletextpad=handletextpad,
)

# save
fig.savefig(os.path.join(plot_path, "performance_metrics_sample-and-feature-wise.png"), bbox_inches="tight", dpi=300)