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
figure_height = 12
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
n_subplots_cols = 5
scaling_cols = 4
n_cols = n_subplots_cols * scaling_cols
n_rows = 10
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=0.0, hspace=0.0)
ax_list = []
palette_models = ["#DAA327", "#015799"]
stage_palette = "magma_r"
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

for i, batch in enumerate(stages):
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
    # rep_default = rep_default.sample(frac=1)
    rep_sup = pd.DataFrame(rep_sup, columns=["D1", "D2"])
    rep_sup["batch"] = hue_gast
    # rep_sup = rep_sup.sample(frac=1)
    # plot them
    ax_list.append(fig.add_subplot(gs[0:2, (i*scaling_cols):((i+1)*scaling_cols)]))
    if i == 0:
        ax_list[-1].text(
            grid_letter_positions[0] - 0.025,
            1.0 + grid_letter_positions[1],
            "A",
            transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize,
            va="bottom",
            fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight,
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

    ax_list.append(fig.add_subplot(gs[2:4, (i*scaling_cols):((i+1)*scaling_cols)]))
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
    if i < len(stages) - 1:
        ax_list[-2].get_legend().remove()
        ax_list[-1].get_legend().remove()
    else:
        ax_list[-2].legend(
            title="Stage",
            bbox_to_anchor=(1.05, 1.0),
            loc=2,
            borderaxespad=0.0,
            frameon=False,
            handletextpad=handletextpad,
            markerscale=handlesize,
        )
        ax_list[-1].get_legend().remove()
    # remove axis labels and ticks
    if i > 0:
        #ax_list[-1].set_xlabel("")
        ax_list[-1].set_ylabel("")
        #ax_list[-2].set_xlabel("")
        ax_list[-2].set_ylabel("")
    else:
        #ax_list[-1].set_xlabel("")
        #ax_list[-2].set_xlabel("")
        ax_list[-1].set_ylabel("supervised")
        ax_list[-2].set_ylabel("naive")
    ax_list[-1].set_xlabel("left out: {}".format(batch))
    if i == 2:
        ax_list[-2].set_title("Covariate test representations")
    # remove all axis ticks
    ax_list[-1].set_xticks([])
    ax_list[-1].set_yticks([])
    ax_list[-2].set_xticks([])
    ax_list[-2].set_yticks([])


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

for i, batch in enumerate(batches):
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
    # rep_default = rep_default.sample(frac=1)
    rep_sup = pd.DataFrame(rep_sup, columns=["D1", "D2"])
    rep_sup["batch"] = hue_gast
    # rep_sup = rep_sup.sample(frac=1)
    # plot them
    start = (n_subplots_cols-scaling_cols) * int(scaling_cols/2) + i*scaling_cols
    end = (n_subplots_cols-scaling_cols) * int(scaling_cols/2) + (i+1)*scaling_cols
    ax_list.append(fig.add_subplot(gs[6:8, start:end]))
    if i == 0:
        ax_list[-1].text(
            grid_letter_positions[0] - 0.535,
            1.0 + grid_letter_positions[1],
            "B",
            transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize,
            va="bottom",
            fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight,
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

    ax_list.append(fig.add_subplot(gs[8:10, start:end]))
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
    if i < len(batches) - 1:
        ax_list[-2].get_legend().remove()
        ax_list[-1].get_legend().remove()
    else:
        ax_list[-2].legend(
            title="Site",
            bbox_to_anchor=(1.05, 1.0),
            loc=2,
            borderaxespad=0.0,
            frameon=False,
            handletextpad=handletextpad,
            markerscale=handlesize,
        )
        ax_list[-1].get_legend().remove()
    # remove axis labels and ticks
    if i > 0:
        #ax_list[-1].set_xlabel("")
        ax_list[-1].set_ylabel("")
        #ax_list[-2].set_xlabel("")
        ax_list[-2].set_ylabel("")
    else:
        #ax_list[-1].set_xlabel("")
        #ax_list[-2].set_xlabel("")
        ax_list[-1].set_ylabel("supervised")
        ax_list[-2].set_ylabel("naive")
    ax_list[-1].set_xlabel("left out: {}".format(batch))
    if i == 2:
        #ax_list[-2].set_title("Covariate test representations", loc="left")
        ax_list[-2].text(
            -0.4,
            1.95,
            "Covariate test representations",
            transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize-1,
            va="bottom",
            fontfamily=grid_letter_fontfamily,
        )
    # remove all axis ticks
    ax_list[-1].set_xticks([])
    ax_list[-1].set_yticks([])
    ax_list[-2].set_xticks([])
    ax_list[-2].set_yticks([])

#####################
# save
#####################

plt.savefig(
    plot_dir + "plots/main/fig3_second_part_supp.png", dpi=720, bbox_inches="tight"
)