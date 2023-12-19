"""
This script analyses the effect on prediction and data integration
of leaving a batch out of training
"""

# imports
import pandas as pd
import numpy as np
import anndata as ad
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000

from omicsdgd import DGD
from omicsdgd.functions._analysis import testset_reconstruction_evaluation_extended, compute_expression_error, binary_output_scores

####################
# collect test errors per model and sample
####################
# load data
save_dir = "../results/trained_models/"
data_name = "human_bonemarrow"
adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
adata.X = adata.layers["counts"]
train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
test_indices = list(np.where(adata.obs["train_val_test"] == "test")[0])
trainset = adata[train_indices, :].copy()
testset = adata[test_indices, :].copy()
sampling_indices = np.random.choice(np.arange(testset.X.shape[0]), 100)
batches = trainset.obs["Site"].unique()
modality_switch = 13431
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

# make sure the results directory exists
import os
result_path = "../results/revision/analysis/batch_integration/" + data_name + "/"
plot_path = "../results/revision/plots/" + data_name + "/"
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def calculate_mean_errors(test_errors, testset, modality_switch):
    # calculate the average error per sample and modality, along with standard error
    test_errors_rna = test_errors[:, : modality_switch].detach().cpu().numpy()
    test_errors_atac = test_errors[:, modality_switch :].detach().cpu().numpy()
    test_errors_rna_mean_sample = np.mean(test_errors_rna, axis=1)
    test_errors_atac_mean_sample = np.mean(test_errors_atac, axis=1)
    test_errors_rna_std_sample = np.std(test_errors_rna, axis=1)
    test_errors_atac_std_sample = np.std(test_errors_atac, axis=1)
    test_errors_rna_se_sample = test_errors_rna_std_sample / np.sqrt(test_errors_rna.shape[1])
    test_errors_atac_se_sample = test_errors_atac_std_sample / np.sqrt(test_errors_atac.shape[1])
    # now also gene-wise
    test_errors_rna_mean_gene = np.mean(test_errors_rna, axis=0)
    test_errors_atac_mean_gene = np.mean(test_errors_atac, axis=0)
    test_errors_rna_std_gene = np.std(test_errors_rna, axis=0)
    test_errors_atac_std_gene = np.std(test_errors_atac, axis=0)
    test_errors_rna_se_gene = test_errors_rna_std_gene / np.sqrt(test_errors_rna.shape[0])
    test_errors_atac_se_gene = test_errors_atac_std_gene / np.sqrt(test_errors_atac.shape[0])
    # save the results
    df1 = pd.DataFrame(
        {
            "rna_mean": test_errors_rna_mean_sample,
            "rna_std": test_errors_rna_std_sample,
            "rna_se": test_errors_rna_se_sample,
            "atac_mean": test_errors_atac_mean_sample,
            "atac_std": test_errors_atac_std_sample,
            "atac_se": test_errors_atac_se_sample,
            "batch": testset.obs["Site"].values,
        }
    )
    df2_temp = pd.DataFrame(
        {
            "rna_mean": test_errors_rna_mean_gene,
            "rna_std": test_errors_rna_std_gene,
            "rna_se": test_errors_rna_se_gene,
            "feature": testset.var_names[:modality_switch],
            "modality": "rna"
        }
    )
    df2 = pd.concat(
        [
            df2_temp,
            pd.DataFrame(
                {
                    "atac_mean": test_errors_atac_mean_gene,
                    "atac_std": test_errors_atac_std_gene,
                    "atac_se": test_errors_atac_se_gene,
                    "feature": testset.var_names[modality_switch:],
                    "modality": "atac"
                }
            )
        ]
    )
    return df1, df2

#"""
# loop over models, make predictions and compute errors per test sample
batches_left_out = ["none", "site1", "site2", "site3", "site4"]
model_names = [
    "human_bonemarrow_l20_h2-3_test50e",
    "human_bonemarrow_l20_h2-3_leftout_site1_test50e_default",
    "human_bonemarrow_l20_h2-3_leftout_site2_test50e_default",
    "human_bonemarrow_l20_h2-3_leftout_site3_test50e_default",
    "human_bonemarrow_l20_h2-3_leftout_site4_test50e_default",
]
for i, model_name in enumerate(model_names):
    print("batch left out:",batches_left_out[i])
    if batches_left_out[i] != "none":
        train_indices = [
            x
            for x in np.arange(len(trainset))
            if trainset.obs["Site"].values[x] != batches_left_out[i]
        ]
        model = DGD.load(
            data=trainset[train_indices],
            save_dir=save_dir + data_name + "/",
            model_name=model_name,
        )
    else:
        model = DGD.load(
            data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
        )
    model.init_test_set(testset)

    # get test predictions
    print("   predicting test samples")
    test_predictions = model.predict_from_representation(
        model.test_rep, model.correction_test_rep
    )
    metrics_temp = testset_reconstruction_evaluation_extended(
        testset, model, modality_switch, library, thresholds=[0.2]
    )
    # save
    metrics_temp.to_csv(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_recon_metrics_default.csv"
    )
    rmse_sample = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_sample', reduction="sample")
    tpr_s, tnr_s, ba_s, _, _ = binary_output_scores(testset, model, library[:,1].unsqueeze(1), modality_switch, threshold=0.2, reduction="sample", return_all=True)
    df_sample = pd.DataFrame(
        {
            "rmse": rmse_sample.detach().cpu().numpy(),
            "tpr": tpr_s.detach().cpu().numpy(),
            "tnr": tnr_s.detach().cpu().numpy(),
            "ba": ba_s.detach().cpu().numpy(),
            "batch": testset.obs["Site"].values,
        }
    )
    df_sample.to_csv(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_RMSE-BA_samplewise_default.csv"
    )
    rmse_feature = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_feature', reduction="feature")
    tpr_f, tnr_f, ba_f, _, _ = binary_output_scores(testset, model, library[:,1].unsqueeze(1), modality_switch, threshold=0.2, reduction="feature", return_all=True)
    df_feature = pd.DataFrame(
        {
            "rmse": rmse_feature.detach().cpu().numpy(),
            "feature_name": testset.var_names[:modality_switch],
            "feature": "rna"
        }
    )
    df_feature = pd.concat(
        [
            df_feature,
            pd.DataFrame(
                {
                    "tpr": tpr_f.detach().cpu().numpy(),
                    "tnr": tnr_f.detach().cpu().numpy(),
                    "ba": ba_f.detach().cpu().numpy(),
                    "feature_name": testset.var_names[modality_switch:],
                    "feature": "atac"
                }
            )
        ]
    )
    df_feature.to_csv(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_RMSE-BA_genewise_default.csv"
    )
    # plot the test predictions against the true values (for rna and atac separately in 2 subplots)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #print(test_predictions[0].detach().cpu().numpy().flatten().shape)
    #print(test_predictions[0].detach().cpu()[sampling_indices,:].numpy().flatten().shape)
    #print("...")
    #print(test_predictions[0].detach().cpu().shape)
    #print(library[:,0].shape)
    #print((test_predictions[0].detach().cpu()*library[:,0]).shape)
    # plot a black line for y=x
    ax[0].plot(
        np.squeeze(np.asarray(testset.X[:, :modality_switch].todense())[sampling_indices,:]).flatten(),
        np.squeeze(np.asarray(testset.X[:, :modality_switch].todense())[sampling_indices,:]).flatten(),
        color="black",
        linewidth=0.5,
    )
    ax[0].scatter(
        np.squeeze(np.asarray(testset.X[:, :modality_switch].todense())[sampling_indices,:]).flatten(),
        (test_predictions[0].detach().cpu()*library[:,0].unsqueeze(1)).numpy()[sampling_indices,:].flatten(),
        s=1,
        alpha=0.1,
    )
    ax[0].set_ylabel("predicted")
    ax[0].set_xlabel("true")
    ax[0].set_title("RNA")
    # change the scale to log
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    # plot a black line for y=x
    ax[1].plot(
        np.squeeze(np.asarray(testset.X[:, modality_switch:].todense())[sampling_indices,:]).flatten(),
        np.squeeze(np.asarray(testset.X[:, modality_switch:].todense())[sampling_indices,:]).flatten(),
        color="black",
        linewidth=0.5,
    )
    ax[1].scatter(
        np.squeeze(np.asarray(testset.X[:, modality_switch:].todense())[sampling_indices,:]).flatten(),
        (test_predictions[1].detach().cpu()*library[:,1].unsqueeze(1)).numpy()[sampling_indices,:].flatten(),
        s=1,
        alpha=0.1,
    )
    ax[1].set_ylabel("predicted")
    ax[1].set_xlabel("true")
    ax[1].set_title("ATAC")
    # change the scale to log
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    fig.savefig(
        plot_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_recon_scatter_default.png",
        dpi=300,
        bbox_inches="tight"
    )
    # save test predictions as numpy
    print("   computing test errors")
    test_errors = model.get_prediction_errors(
        test_predictions, model.test_set, reduction="gene"
    )
    test_predictions = None
    df_sample, df_gene = calculate_mean_errors(test_errors, testset, modality_switch)
    df_sample.to_csv(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_errors_samplewise_default.csv"
    )
    df_gene.to_csv(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_errors_genewise_default.csv"
    )
    model = None
    test_errors = None
print("saved dgd predictions")

#exit()

####################

batches_left_out = ["site1", "site2", "site3", "site4"]
model_names = [
    "human_bonemarrow_l20_h2-3_leftout_site1_test100e_covSupervised_beta10",
    "human_bonemarrow_l20_h2-3_leftout_site2_test100e_covSupervised_beta10",
    "human_bonemarrow_l20_h2-3_leftout_site3_test100e_covSupervised_beta10",
    "human_bonemarrow_l20_h2-3_leftout_site4_test100e_covSupervised_beta10",
]
for i, model_name in enumerate(model_names):
    print("batch left out:",batches_left_out[i])
    if batches_left_out[i] != "none":
        train_indices = [
            x
            for x in np.arange(len(trainset))
            if trainset.obs["Site"].values[x] != batches_left_out[i]
        ]
        model = DGD.load(
            data=trainset[train_indices],
            save_dir=save_dir + data_name + "/",
            model_name=model_name,
        )
    else:
        model = DGD.load(
            data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
        )
    model.init_test_set(testset)

    # get test predictions
    print("   predicting test samples")
    test_predictions = model.predict_from_representation(
        model.test_rep, model.correction_test_rep
    )
    metrics_temp = testset_reconstruction_evaluation_extended(
        testset, model, modality_switch, library, thresholds=[0.2]
    )
    # save
    metrics_temp.to_csv(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_recon_metrics_supervised.csv"
    )
    rmse_sample = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_sample', reduction="sample")
    tpr_s, tnr_s, ba_s, _, _ = binary_output_scores(testset, model, library[:,1].unsqueeze(1), modality_switch, threshold=0.2, reduction="sample", return_all=True)
    df_sample = pd.DataFrame(
        {
            "rmse": rmse_sample.detach().cpu().numpy(),
            "tpr": tpr_s.detach().cpu().numpy(),
            "tnr": tnr_s.detach().cpu().numpy(),
            "ba": ba_s.detach().cpu().numpy(),
            "batch": testset.obs["Site"].values,
        }
    )
    df_sample.to_csv(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_RMSE-BA_samplewise_supervised.csv"
    )
    rmse_feature = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_feature', reduction="feature")
    tpr_f, tnr_f, ba_f, _, _ = binary_output_scores(testset, model, library[:,1].unsqueeze(1), modality_switch, threshold=0.2, reduction="feature", return_all=True)
    df_feature = pd.DataFrame(
        {
            "rmse": rmse_feature.detach().cpu().numpy(),
            "feature_name": testset.var_names[:modality_switch],
            "feature": "rna"
        }
    )
    df_feature = pd.concat(
        [
            df_feature,
            pd.DataFrame(
                {
                    "tpr": tpr_f.detach().cpu().numpy(),
                    "tnr": tnr_f.detach().cpu().numpy(),
                    "ba": ba_f.detach().cpu().numpy(),
                    "feature_name": testset.var_names[modality_switch:],
                    "feature": "atac"
                }
            )
        ]
    )
    df_feature.to_csv(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_RMSE-BA_genewise_supervised.csv"
    )
    # plot the test predictions against the true values (for rna and atac separately in 2 subplots)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(
        np.squeeze(np.asarray(testset.X[:, :modality_switch].todense())[sampling_indices,:]).flatten(),
        np.squeeze(np.asarray(testset.X[:, :modality_switch].todense())[sampling_indices,:]).flatten(),
        color="black",
        linewidth=0.5,
    )
    ax[0].scatter(
        np.squeeze(np.asarray(testset.X[:, :modality_switch].todense())[sampling_indices,:]).flatten(),
        (test_predictions[0].detach().cpu()*library[:,0].unsqueeze(1)).numpy()[sampling_indices,:].flatten(),
        s=1,
        alpha=0.1,
    )
    ax[0].set_ylabel("predicted")
    ax[0].set_xlabel("true")
    ax[0].set_title("RNA")
    # change the scale to log
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    # plot a black line for y=x
    ax[1].plot(
        np.squeeze(np.asarray(testset.X[:, modality_switch:].todense())[sampling_indices,:]).flatten(),
        np.squeeze(np.asarray(testset.X[:, modality_switch:].todense())[sampling_indices,:]).flatten(),
        color="black",
        linewidth=0.5,
    )
    ax[1].scatter(
        np.squeeze(np.asarray(testset.X[:, modality_switch:].todense())[sampling_indices,:]).flatten(),
        (test_predictions[1].detach().cpu()*library[:,1].unsqueeze(1)).numpy()[sampling_indices,:].flatten(),
        s=1,
        alpha=0.1,
    )
    ax[1].set_ylabel("predicted")
    ax[1].set_xlabel("true")
    ax[1].set_title("ATAC")
    # change the scale to log
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    fig.savefig(
        plot_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_recon_scatter_supervised.png",
        dpi=300,
        bbox_inches="tight"
    )
    # save test predictions as numpy
    # get test errors
    print("   computing test errors")
    test_errors = model.get_prediction_errors(
        test_predictions, model.test_set, reduction="gene"
    )
    test_predictions = None
    df_sample, df_gene = calculate_mean_errors(test_errors, testset, modality_switch)
    df_sample.to_csv(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_errors_samplewise_supervised.csv"
    )
    df_gene.to_csv(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_errors_genewise_supervised.csv"
    )
    model = None
    test_predictions = None
    test_errors = None
print("saved dgd predictions")
#"""


# now do the three models with 75% of the data
random_seeds = [0, 37, 8790]
model_name = "human_bonemarrow_l20_h2-3_75percent_rs"

for i, rs in enumerate(random_seeds):
    np.random.seed(0)
    indices_keep = np.random.choice(
        np.arange(len(trainset)), int(0.75 * trainset.n_obs), replace=False
    )
    train_indices = list(np.where(trainset.obs["train_val_test"] == "train")[0])
    model = DGD.load(
            data=trainset[train_indices],
            save_dir=save_dir + data_name + "/",
            model_name=model_name + str(rs) + "_test50e",
        )
    model.init_test_set(testset)

    # get test predictions
    print("   predicting test samples")
    test_predictions = model.predict_from_representation(
        model.test_rep, model.correction_test_rep
    )
    metrics_temp = testset_reconstruction_evaluation_extended(
        testset, model, modality_switch, library, thresholds=[0.2]
    )
    # save
    metrics_temp.to_csv(
        result_path
        + data_name
        + "_75percent_rs" + str(rs)
        + "_recon_metrics.csv"
    )
    rmse_sample = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_sample', reduction="sample")
    tpr_s, tnr_s, ba_s, _, _ = binary_output_scores(testset, model, library[:,1].unsqueeze(1), modality_switch, threshold=0.2, reduction="sample", return_all=True)
    df_sample = pd.DataFrame(
        {
            "rmse": rmse_sample.detach().cpu().numpy(),
            "tpr": tpr_s.detach().cpu().numpy(),
            "tnr": tnr_s.detach().cpu().numpy(),
            "ba": ba_s.detach().cpu().numpy(),
            "batch": testset.obs["Site"].values,
        }
    )
    df_sample.to_csv(
        result_path
        + data_name
        + "_75percent_rs" + str(rs)
        + "_RMSE-BA_samplewise.csv"
    )
    rmse_feature = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_feature', reduction="feature")
    tpr_f, tnr_f, ba_f, _, _ = binary_output_scores(testset, model, library[:,1].unsqueeze(1), modality_switch, threshold=0.2, reduction="feature", return_all=True)
    df_feature = pd.DataFrame(
        {
            "rmse": rmse_feature.detach().cpu().numpy(),
            "feature_name": testset.var_names[:modality_switch],
            "feature": "rna"
        }
    )
    df_feature = pd.concat(
        [
            df_feature,
            pd.DataFrame(
                {
                    "tpr": tpr_f.detach().cpu().numpy(),
                    "tnr": tnr_f.detach().cpu().numpy(),
                    "ba": ba_f.detach().cpu().numpy(),
                    "feature_name": testset.var_names[modality_switch:],
                    "feature": "atac"
                }
            )
        ]
    )
    df_feature.to_csv(
        result_path
        + data_name
        + "_75percent_rs" + str(rs)
        + "_RMSE-BA_genewise.csv"
    )
    # plot the test predictions against the true values (for rna and atac separately in 2 subplots)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(
        np.squeeze(np.asarray(testset.X[:, :modality_switch].todense())[sampling_indices,:]).flatten(),
        np.squeeze(np.asarray(testset.X[:, :modality_switch].todense())[sampling_indices,:]).flatten(),
        color="black",
        linewidth=0.5,
    )
    ax[0].scatter(
        np.squeeze(np.asarray(testset.X[:, :modality_switch].todense())[sampling_indices,:]).flatten(),
        (test_predictions[0].detach().cpu()*library[:,0].unsqueeze(1)).numpy()[sampling_indices,:].flatten(),
        s=1,
        alpha=0.1,
    )
    ax[0].set_ylabel("predicted")
    ax[0].set_xlabel("true")
    ax[0].set_title("RNA")
    # change the scale to log
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    # plot a black line for y=x
    ax[1].plot(
        np.squeeze(np.asarray(testset.X[:, modality_switch:].todense())[sampling_indices,:]).flatten(),
        np.squeeze(np.asarray(testset.X[:, modality_switch:].todense())[sampling_indices,:]).flatten(),
        color="black",
        linewidth=0.5,
    )
    ax[1].scatter(
        np.squeeze(np.asarray(testset.X[:, modality_switch:].todense())[sampling_indices,:]).flatten(),
        (test_predictions[1].detach().cpu()*library[:,1].unsqueeze(1)).numpy()[sampling_indices,:].flatten(),
        s=1,
        alpha=0.1,
    )
    ax[1].set_ylabel("predicted")
    ax[1].set_xlabel("true")
    ax[1].set_title("ATAC")
    # change the scale to log
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    fig.savefig(
        plot_path
        + data_name
        + "_75percent_rs" + str(rs)
        + "_recon_scatter.png",
        dpi=300,
        bbox_inches="tight"
    )
    # save test predictions as numpy
    # get test errors
    print("   computing test errors")
    test_errors = model.get_prediction_errors(
        test_predictions, model.test_set, reduction="gene"
    )
    test_predictions = None
    df_sample, df_gene = calculate_mean_errors(test_errors, testset, modality_switch)
    df_sample.to_csv(
        result_path
        + data_name
        + "_75percent_rs" + str(rs)
        + "_errors_samplewise.csv"
    )
    df_gene.to_csv(
        result_path
        + data_name
        + "_75percent_rs" + str(rs)
        + "_errors_genewise.csv"
    )
    model = None
    test_predictions = None
    test_errors = None
print("saved dgd predictions")