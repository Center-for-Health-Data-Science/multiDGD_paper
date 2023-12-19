"""
This script analyses the effect on prediction and data integration
of leaving a batch out of training
"""

# imports
import pandas as pd
import numpy as np
import anndata as ad

from omicsdgd import DGD

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
batches = trainset.obs["Site"].unique()

# make sure the results directory exists
import os
result_path = "../results/revision/analysis/batch_integration/" + data_name + "/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# loop over models, make predictions and compute errors per test sample
batches_left_out = ["none", "site1", "site2", "site3", "site4"]
model_names = [
    "human_bonemarrow_l20_h2-3",
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
    # extract the correction test reps and gmm means
    cov_rep = model.correction_test_rep.z.detach().cpu().numpy()
    cov_means = model.correction_gmm.mean.detach().cpu().numpy()
    # save the correction test reps and gmm means
    np.save(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_covariate_representations_default.npy",
        cov_rep,
    )
    np.save(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_covariate_means_default.npy",
        cov_means,
    )
    model = None
print("saved reps")


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
    # extract the correction test reps and gmm means
    cov_rep = model.correction_test_rep.z.detach().cpu().numpy()
    cov_means = model.correction_gmm.mean.detach().cpu().numpy()
    # save the correction test reps and gmm means
    np.save(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_covariate_representations_supervised.npy",
        cov_rep,
    )
    np.save(
        result_path
        + data_name
        + "_"
        + batches_left_out[i]
        + "_covariate_means_supervised.npy",
        cov_means,
    )
    model = None
print("saved reps")