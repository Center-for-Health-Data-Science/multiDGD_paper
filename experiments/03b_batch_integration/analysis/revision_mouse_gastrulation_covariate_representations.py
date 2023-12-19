"""
This script analyses the effect on prediction and data integration
of leaving a batch out of training
"""

# imports
import pandas as pd
import numpy as np
import mudata as md

from omicsdgd import DGD

####################
# collect test errors per model and sample
####################
# load data
save_dir = "../results/trained_models/"
data_name = "mouse_gastrulation"
mudata = md.read("../../data/mouse_gastrulation.h5mu", backed=False)
train_indices = list(np.where(mudata.obs["train_val_test"] == "train")[0])
test_indices = list(np.where(mudata.obs["train_val_test"] == "test")[0])
trainset = mudata[train_indices, :].copy()
testset = mudata[test_indices, :].copy()
batches = trainset.obs["stage"].unique()

# make sure the results directory exists
import os
result_path = "../results/revision/analysis/batch_integration/" + data_name + "/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

#"""
# loop over models, make predictions and compute errors per test sample
batches_left_out = ["none", "E7.5", "E7.75", "E8.0", "E8.5", "E8.75"]
model_names = [
    "mouse_gast_l20_h2-2_rs0",
    "mouse_gast_l20_h2-2_rs0_leftout_E7.5_test50e_default",
    "mouse_gast_l20_h2-2_rs0_leftout_E7.75_test50e_default",
    "mouse_gast_l20_h2-2_rs0_leftout_E8.0_test50e_default",
    "mouse_gast_l20_h2-2_rs0_leftout_E8.5_test50e_default",
    "mouse_gast_l20_h2-2_rs0_leftout_E8.75_test50e_default"
]
for i, model_name in enumerate(model_names):
    print("batch left out:",batches_left_out[i])
    if batches_left_out[i] != "none":
        train_indices = [
            x
            for x in np.arange(len(trainset))
            if trainset.obs["stage"].values[x] != batches_left_out[i]
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

#exit()
#"""

####################

batches_left_out = ["E7.5", "E7.75", "E8.0", "E8.5", "E8.75"]
model_names = [
    "mouse_gast_l20_h2-2_rs0_leftout_E7.5_test100e_covSupervised_beta10",
    "mouse_gast_l20_h2-2_rs0_leftout_E7.75_test100e_covSupervised_beta5_finetuned",
    "mouse_gast_l20_h2-2_rs0_leftout_E8.0_test100e_covSupervised_beta5_finetuned",
    "mouse_gast_l20_h2-2_rs0_leftout_E8.5_test100e_covSupervised_beta10",
    "mouse_gast_l20_h2-2_rs0_leftout_E8.75_test100e_covSupervised_beta20"
]
for i, model_name in enumerate(model_names):
    print("batch left out:",batches_left_out[i])
    if batches_left_out[i] != "none":
        train_indices = [
            x
            for x in np.arange(len(trainset))
            if trainset.obs["stage"].values[x] != batches_left_out[i]
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