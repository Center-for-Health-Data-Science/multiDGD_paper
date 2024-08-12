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
from omicsdgd.functions._analysis import discrete_kullback_leibler

#####################
# define model names, directory and batches
#####################
save_dir = "../results/trained_models/"
analysis_dir = "../results/revision/analysis/"
plot_dir = "../results/revision/"

#####################
#####################
# set up figure
#####################
#####################
figure_height = 16
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
n_cols = 10
n_rows = 9
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=0.0, hspace=0.0)
ax_list = []
palette_models = ["#DAA327", "#015799"]
stage_palette = "magma_r"
# make a discrete stage pallete with 5 stages
stage_palette = sns.color_palette("magma_r", 5)
palette = ["#FFD6D6", "#EC6B6B", "#015799"]
batch_palette = ["#EEE7A8", "cornflowerblue", "darkmagenta", "darkslategray"]
plt.rcParams.update(
    {
        "font.size": 6,
        "axes.linewidth": 0.5,
        "xtick.major.size": 1.5,
        "xtick.major.width": 0.5,
        "ytick.major.size": 1.5,
        "ytick.major.width": 0.5,
    }
)
handletextpad = 0.1
legend_x_dist, legend_y_dist = -0.02, 0.0
grid_letter_positions = [-0.1, 0.1]
grid_letter_fontsize = 8
grid_letter_fontfamily = "sans-serif"
grid_letter_fontweight = "bold"
point_size = 0.5
line_width = 0.5
strip_size = 3
box_width = 0.5
point_linewidth = 0.0
handlesize = 0.3
alpha = 0.1
# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
# change

#####################
# import graphic
#####################

graphic_dir = "../results/revision/graphics/multidgd_paper_cov_modelling.png"

ax_list.append(fig.add_subplot(gs[0:3, 0:4]))
ax_list[-1].imshow(plt.imread(graphic_dir))
ax_list[-1].axis("off")
ax_list[-1].text(
    grid_letter_positions[0] - 0.155,
    1.05,
    "A",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
ax_list[-1].set_title("Inference of novel covariates   ")

#####################
# get data
#####################

# first, access the predictions for the normal models
data_name = "mouse_gastrulation"
n_features_mg = 11792+69862
stages = ["E7.5", "E7.75", "E8.0", "E8.5", "E8.75"]

predictions_gast_original = pd.read_csv(os.path.join(analysis_dir, "batch_integration/{}/{}_none_prediction_errors_default.csv".format(data_name, data_name)))
hue_gast = predictions_gast_original["batch_id"]
predictions_gast_original["prediction_type"] = "naive (trained on all)"
pred_default = pd.DataFrame()
pred_sup = pd.DataFrame()
for i, batch in enumerate(stages):
    pred_default_temp = pd.read_csv(
        os.path.join(
            analysis_dir,
            "batch_integration/{}/{}_{}_prediction_errors_default.csv".format(
                data_name, data_name, batch
            ),
        )
    )
    pred_sup_temp = pd.read_csv(
        os.path.join(
            analysis_dir,
            "batch_integration/{}/{}_{}_prediction_errors_supervised.csv".format(
                data_name, data_name, batch
            ),
        )
    )
    pred_default = pd.concat([pred_default, pred_default_temp])
    pred_sup = pd.concat([pred_sup, pred_sup_temp])

# only keep the entries where batch_id and model_id are the same
pred_default = pred_default[pred_default["batch_id"] == pred_default["model_id"]]
pred_default["prediction_type"] = "naive (left-out)"
pred_sup = pred_sup[pred_sup["batch_id"] == pred_sup["model_id"]]
pred_sup["prediction_type"] = "supervised (left-out)"
# combine the dataframes
predictions_gast = pd.concat(
    [predictions_gast_original, pred_default, pred_sup]
)
predictions_gast["error"] = predictions_gast["error"] / n_features_mg

#####################
# process data
#####################

# load the sample-wise errors for the original model
sample_gast_errors = pd.read_csv(os.path.join(analysis_dir, "batch_integration/mouse_gastrulation/mouse_gastrulation_none_errors_samplewise_default.csv"))
sample_gast_errors["model"] = "full"
# loop through the batches and get the sample-wise errors for the left-out batches
for i, batch in enumerate(stages):
    sample_errors_temp = pd.read_csv(os.path.join(analysis_dir, "batch_integration/mouse_gastrulation/mouse_gastrulation_{}_errors_samplewise_default.csv".format(batch)))
    # only keep the entries where batch_id and model_id are the same
    sample_errors_temp = sample_errors_temp[sample_errors_temp["batch"] == batch]
    sample_errors_temp["model"] = "leave-one-out (naive)"
    sample_gast_errors = pd.concat([sample_gast_errors, sample_errors_temp])
    sample_errors_temp = pd.read_csv(os.path.join(analysis_dir, "batch_integration/mouse_gastrulation/mouse_gastrulation_{}_errors_samplewise_supervised.csv".format(batch)))
    # only keep the entries where batch_id and model_id are the same
    sample_errors_temp = sample_errors_temp[sample_errors_temp["batch"] == batch]
    sample_errors_temp["model"] = "leave-one-out (supervised)"
    sample_gast_errors = pd.concat([sample_gast_errors, sample_errors_temp])
# sort the dataframe by model with specific order (as categorical)
sample_gast_errors["model"] = pd.Categorical(sample_gast_errors["model"], ["full", "leave-one-out (naive)", "leave-one-out (supervised)"])
sample_gast_errors['mean_error'] = sample_gast_errors['rna_mean'] + sample_gast_errors['atac_mean']

#####################
# plot
#####################

i = 2
batch = stages[i]
# get the representations (they are stored as numpy arrays)
rep_default = np.load(
    os.path.join(
        analysis_dir,
        "batch_integration/{}/{}_{}_covariate_representations_default.npy".format(
            data_name, data_name, batch
        ),
    )
)
rep_sup = np.load(
    os.path.join(
        analysis_dir,
        "batch_integration/{}/{}_{}_covariate_representations_supervised.npy".format(
            data_name, data_name, batch
        ),
    )
)
# make dataframes out of the reps and randomise the order
rep_default = pd.DataFrame(rep_default, columns=["D1", "D2"])
rep_default["batch"] = hue_gast
# "seen" for samples that were seen during training, "unseen" for batch
rep_default["seen"] = ["unseen" if x == batch else "seen" for x in hue_gast]
# rep_default = rep_default.sample(frac=1)
rep_sup = pd.DataFrame(rep_sup, columns=["D1", "D2"])
rep_sup["batch"] = hue_gast
rep_sup["seen"] = ["unseen" if x == batch else "seen" for x in hue_gast]
# rep_sup = rep_sup.sample(frac=1)
# plot them
ax_list.append(fig.add_subplot(gs[5:7, 0:2]))
ax_list[-1].text(
    grid_letter_positions[0] - 0.0,
    1.15,
    "D",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
ax_list[-1].text(
    0.5,
    1.15,
    "Mouse gastrulation: left out {}".format(batch),
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize-1,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
)
sns.scatterplot(
    data=rep_default,
    x="D1",
    y="D2",
    hue="batch",
    palette=stage_palette,
    alpha=alpha,
    ax=ax_list[-1],
    s=point_size,
    linewidth=point_linewidth,
)
#ax_list[-1].set_title("left out: {}".format(batch))

ax_list.append(fig.add_subplot(gs[5:7, 2:4]))
sns.scatterplot(
    data=rep_sup,
    x="D1",
    y="D2",
    hue="batch",
    palette=stage_palette,
    alpha=alpha,
    ax=ax_list[-1],
    s=point_size,
    linewidth=point_linewidth,
)
# remove the legends unless its the last plot
ax_list[-2].get_legend().remove()
ax_list[-1].legend(
    title="stage",
    bbox_to_anchor=(1.05, 1.0),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad,
    markerscale=handlesize,
)
#ax_list[-1].set_ylabel("")
#ax_list[-2].set_ylabel("mouse gastrulation\nleft out: {}".format(batch))
ax_list[-2].set_title("naive")
ax_list[-1].set_title("supervised")
ax_list[-1].set_xlabel("")
ax_list[-2].set_xlabel("")
ax_list[-2].set_ylabel("D2")
ax_list[-1].set_ylabel("")
# remove all axis ticks
ax_list[-1].set_xticks([])
ax_list[-1].set_yticks([])
ax_list[-2].set_xticks([])
ax_list[-2].set_yticks([])

# sort the dataframes by "seen" (unseen should be on top)
rep_default = rep_default.sort_values(by="seen", ascending=True)
rep_sup = rep_sup.sort_values(by="seen", ascending=True)
ax_list.append(fig.add_subplot(gs[7:9, 0:2]))
sns.scatterplot(
    data=rep_default,
    x="D1",
    y="D2",
    hue="seen",
    palette=["grey", stage_palette[i]],
    alpha=alpha,
    ax=ax_list[-1],
    s=point_size,
    linewidth=point_linewidth,
)
ax_list.append(fig.add_subplot(gs[7:9, 2:4]))
sns.scatterplot(
    data=rep_sup,
    x="D1",
    y="D2",
    hue="seen",
    palette=["grey", stage_palette[i]],
    alpha=alpha,
    ax=ax_list[-1],
    s=point_size,
    linewidth=point_linewidth,
)
# remove the legends unless its the last plot
ax_list[-2].get_legend().remove()
ax_list[-1].legend(
    title="stage",
    bbox_to_anchor=(1.05, 1.0),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad,
    markerscale=handlesize,
)
ax_list[-1].set_xlabel("D1")
ax_list[-2].set_xlabel("D1")
ax_list[-2].set_ylabel("D2")
ax_list[-1].set_ylabel("")
# remove all axis ticks
ax_list[-1].set_xticks([])
ax_list[-1].set_yticks([])
ax_list[-2].set_xticks([])
ax_list[-2].set_yticks([])

###

ax_list.append(fig.add_subplot(gs[0, 6:]))
ax_list[-1].text(
    grid_letter_positions[0] - 0.055,
    1.0 + grid_letter_positions[1],
    "B",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)

sns.pointplot(
    x="batch_id",
    y="error",
    data=predictions_gast,
    hue="prediction_type",
    palette=palette,
    ax=ax_list[-1],
    join=False,
    dodge=0.5,
    markers=".",
    scale=0.5,
    errorbar="se",
    errwidth=0.5,
    capsize=0.2
)
ax_list[-1].set_ylabel("Reconstruction error")
ax_list[-1].set_xlabel("Stage")
handles, labels = ax_list[-1].get_legend_handles_labels()
#ax_list[-1].legend().remove()
ax_list[-1].legend(
    handles,
    ["seen", "unseen\n(naive)", "unseen\n(supervised)"],
    title="integration\nmethod",
    bbox_to_anchor=(1.02, 1),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad * 2,
)
ax_list[-1].set_title("Mouse gastrulation test reconstruction")
ax_list[-1].set_ylim(0.45, 0.95)

ax_list[-1].plot(
    [3.75, 4.25],
    [predictions_gast[predictions_gast['batch_id'] == 'E8.75']['error'].mean() + 0.11] *2,
    color='black',
    linewidth=0.5
)
ax_list[-1].plot(
    [4.0, 4.25],
    [predictions_gast[predictions_gast['batch_id'] == 'E8.75']['error'].mean() + 0.05] *2,
    color='black',
    linewidth=0.5
)
ax_list[-1].text(
    4,
    predictions_gast[predictions_gast['batch_id'] == 'E8.75']['error'].mean() + 0.12,
    "*",
    horizontalalignment='center',
    verticalalignment='center'
)
ax_list[-1].text(
    4.125,
    predictions_gast[predictions_gast['batch_id'] == 'E8.75']['error'].mean() + 0.06,
    "*",
    horizontalalignment='center',
    verticalalignment='center'
)

#####################
# check significance
#####################

# include significance tests
from itertools import combinations
import scipy.stats

models = predictions_gast['prediction_type'].unique()
batches = predictions_gast['batch_id'].unique()
significance_dict = {}

for batch in stages:
    data_batch = predictions_gast[predictions_gast['batch_id'] == batch]
    for model1, model2 in combinations(models, 2):
        group1 = data_batch[data_batch['prediction_type'] == model1]['error']
        group2 = data_batch[data_batch['prediction_type'] == model2]['error']

        # Perform t-test
        #print(scipy.stats.mannwhitneyu(group1, group2))
        _, p_value = scipy.stats.mannwhitneyu(group1, group2)
        significant = p_value < 0.05

        # Store the significance
        if significant:
            significance_dict[(batch, model1, model2)] = p_value
        else:
            significance_dict[(batch, model1, model2)] = significant

# print all entries that are significant
for key, value in significance_dict.items():
    if value:
        print(key, value)

#####################
# bonemarrow
#####################

###
# get data
###

# first, access the predictions for the normal models
data_name = "human_bonemarrow"
n_features_bm = 129921
batches = ["site1", "site2", "site3", "site4"]

###
# process data
###

# load the sample-wise errors for the original model
sample_gast_errors = pd.read_csv(os.path.join(analysis_dir, "batch_integration/{}/{}_none_errors_samplewise_default.csv".format(data_name, data_name)))
sample_gast_errors["model"] = "full"
hue_gast = sample_gast_errors["batch"]
# loop through the batches and get the sample-wise errors for the left-out batches
for i, batch in enumerate(batches):
    sample_errors_temp = pd.read_csv(os.path.join(analysis_dir, "batch_integration/{}/{}_{}_errors_samplewise_default.csv".format(data_name, data_name, batch)))
    # only keep the entries where batch_id and model_id are the same
    sample_errors_temp = sample_errors_temp[sample_errors_temp["batch"] == batch]
    sample_errors_temp["model"] = "leave-one-out (naive)"
    sample_gast_errors = pd.concat([sample_gast_errors, sample_errors_temp])
    sample_errors_temp = pd.read_csv(os.path.join(analysis_dir, "batch_integration/{}/{}_{}_errors_samplewise_supervised.csv".format(data_name, data_name, batch)))
    # only keep the entries where batch_id and model_id are the same
    sample_errors_temp = sample_errors_temp[sample_errors_temp["batch"] == batch]
    sample_errors_temp["model"] = "leave-one-out (supervised)"
    sample_gast_errors = pd.concat([sample_gast_errors, sample_errors_temp])
# sort the dataframe by model with specific order (as categorical)
sample_gast_errors["model"] = pd.Categorical(sample_gast_errors["model"], ["full", "leave-one-out (naive)", "leave-one-out (supervised)"])
sample_gast_errors['mean_error'] = sample_gast_errors['rna_mean'] + sample_gast_errors['atac_mean']


###
# plot
###

i = 0
batch = batches[i]
# get the representations (they are stored as numpy arrays)
rep_default = np.load(
    os.path.join(
        analysis_dir,
        "batch_integration/{}/{}_{}_covariate_representations_default.npy".format(
            data_name, data_name, batch
        ),
    )
)
rep_sup = np.load(
    os.path.join(
        analysis_dir,
        "batch_integration/{}/{}_{}_covariate_representations_supervised.npy".format(
            data_name, data_name, batch
        ),
    )
)
# make dataframes out of the reps and randomise the order
rep_default = pd.DataFrame(rep_default, columns=["D1", "D2"])
rep_default["batch"] = hue_gast
rep_default["seen"] = ["unseen" if x == batch else "seen" for x in hue_gast]
# rep_default = rep_default.sample(frac=1)
rep_sup = pd.DataFrame(rep_sup, columns=["D1", "D2"])
rep_sup["batch"] = hue_gast
rep_sup["seen"] = ["unseen" if x == batch else "seen" for x in hue_gast]
# rep_sup = rep_sup.sample(frac=1)
# plot them
ax_list.append(fig.add_subplot(gs[5:7, 6:8]))
ax_list[-1].text(
    grid_letter_positions[0] - 0.0,
    1.15,
    "E",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)
ax_list[-1].text(
    0.5,
    1.15,
    "Human bone marrow: left out {}".format(batch),
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize-1,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
)
sns.scatterplot(
    data=rep_default,
    x="D1",
    y="D2",
    hue="batch",
    palette=batch_palette,
    alpha=alpha,
    ax=ax_list[-1],
    s=point_size,
    linewidth=point_linewidth,
)
#ax_list[-1].set_title("left out: {}".format(batch))

ax_list.append(fig.add_subplot(gs[5:7, 8:10]))
sns.scatterplot(
    data=rep_sup,
    x="D1",
    y="D2",
    hue="batch",
    palette=batch_palette,
    alpha=alpha,
    ax=ax_list[-1],
    s=point_size,
    linewidth=point_linewidth,
)
# remove the legends unless its the last plot
ax_list[-1].legend(
    title="site",
    bbox_to_anchor=(1.05, 1.0),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad,
    markerscale=handlesize,
)
ax_list[-2].get_legend().remove()
#ax_list[-1].set_ylabel("")
#ax_list[-2].set_ylabel("human bone marrow\nleft out: {}".format(batch))
ax_list[-2].set_title("naive")
ax_list[-1].set_title("supervised")
ax_list[-1].set_xlabel("D1")
ax_list[-2].set_xlabel("D1")
ax_list[-1].set_ylabel("")
ax_list[-2].set_ylabel("D2")
# remove all axis ticks
ax_list[-1].set_xticks([])
ax_list[-1].set_yticks([])
ax_list[-2].set_xticks([])
ax_list[-2].set_yticks([])

# sort the dataframes by "seen" (unseen should be on top)
rep_default = rep_default.sort_values(by="seen", ascending=True)
rep_sup = rep_sup.sort_values(by="seen", ascending=True)
ax_list.append(fig.add_subplot(gs[7:9, 6:8]))
sns.scatterplot(
    data=rep_default,
    x="D1",
    y="D2",
    hue="seen",
    palette=["grey", batch_palette[i]],
    alpha=alpha,
    ax=ax_list[-1],
    s=point_size,
    linewidth=point_linewidth,
)
ax_list.append(fig.add_subplot(gs[7:9, 8:10]))
sns.scatterplot(
    data=rep_sup,
    x="D1",
    y="D2",
    hue="seen",
    palette=["grey", batch_palette[i]],
    alpha=alpha,
    ax=ax_list[-1],
    s=point_size,
    linewidth=point_linewidth,
)
# remove the legends unless its the last plot
ax_list[-2].get_legend().remove()
ax_list[-1].legend(
    title="site",
    bbox_to_anchor=(1.05, 1.0),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad,
    markerscale=handlesize,
)
ax_list[-1].set_xlabel("D1")
ax_list[-2].set_xlabel("D1")
ax_list[-2].set_ylabel("D2")
ax_list[-1].set_ylabel("")
# remove all axis ticks
ax_list[-1].set_xticks([])
ax_list[-1].set_yticks([])
ax_list[-2].set_xticks([])
ax_list[-2].set_yticks([])


ax_list.append(fig.add_subplot(gs[2, 6:]))
ax_list[-1].text(
    grid_letter_positions[0] - 0.055,
    1.0 + grid_letter_positions[1],
    "C",
    transform=ax_list[-1].transAxes + trans,
    fontsize=grid_letter_fontsize,
    va="bottom",
    fontfamily=grid_letter_fontfamily,
    fontweight=grid_letter_fontweight,
)

sns.pointplot(
    x="batch",
    y="mean_error",
    data=sample_gast_errors,
    hue="model",
    palette=palette,
    ax=ax_list[-1],
    join=False,
    dodge=0.5,
    markers=".",
    scale=0.5,
    errorbar="se",
    errwidth=0.5,
    capsize=0.2
)
ax_list[-1].set_ylabel("Reconstruction error")
ax_list[-1].set_xlabel("Site")
handles, labels = ax_list[-1].get_legend_handles_labels()
ax_list[-1].legend().remove()
"""
ax_list[-1].legend(
    handles,
    ["seen", "unseen\n(naive)", "unseen\n(supervised)"],
    title="integration\nmethod",
    bbox_to_anchor=(1.02, 1),
    loc=2,
    borderaxespad=0.0,
    frameon=False,
    handletextpad=handletextpad * 2,
)
"""
ax_list[-1].set_title("Human bone marrow test reconstruction")
ax_list[-1].set_ylim(0.37, 0.56)

ax_list[-1].plot(
    [-0.25, 0.25],
    [sample_gast_errors[sample_gast_errors['batch'] == 'site1']['mean_error'].mean() + 0.022] *2,
    color='black',
    linewidth=0.5
)
ax_list[-1].text(
    0,
    sample_gast_errors[sample_gast_errors['batch'] == 'site1']['mean_error'].mean() + 0.03,
    "*",
    horizontalalignment='center',
    verticalalignment='center'
)

# include significance tests
from itertools import combinations
import scipy.stats

models = sample_gast_errors['model'].unique()
batches = sample_gast_errors['batch'].unique()
significance_dict = {}

print("bonemarrow")
for batch in batches:
    data_batch = sample_gast_errors[sample_gast_errors['batch'] == batch]
    for model1, model2 in combinations(models, 2):
        group1 = data_batch[data_batch['model'] == model1]['mean_error']
        group2 = data_batch[data_batch['model'] == model2]['mean_error']

        # Perform t-test
        #print(scipy.stats.mannwhitneyu(group1, group2))
        _, p_value = scipy.stats.mannwhitneyu(group1, group2)
        significant = p_value < 0.1

        # Store the significance
        if significant:
            significance_dict[(batch, model1, model2)] = p_value
        else:
            significance_dict[(batch, model1, model2)] = significant

# print all entries that are significant
for key, value in significance_dict.items():
    if value:
        print(key, value)

#####################
# save
#####################

plt.savefig(
    plot_dir + "plots/main/fig3_second_part_v2.png", dpi=720, bbox_inches="tight"
)