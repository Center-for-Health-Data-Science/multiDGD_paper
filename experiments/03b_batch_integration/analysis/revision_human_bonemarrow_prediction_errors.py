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
result_path = "../results/revision/analysis/batch_integration/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

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
    # get test errors
    print("   computing test errors")
    test_errors = model.get_prediction_errors(
        test_predictions, model.test_set, reduction="sample"
    )

    ###
    # collect relevant errors and save in meaningfully plottable dataframe
    ###
    # make a dataframe per model with the following columns:
    # - sample id
    # - batch id of sample
    # - error of sample
    # - model id (in terms of batch left out)
    print("   collecting results")
    temp_df = pd.DataFrame(
        {
            "sample_id": testset.obs.index,
            "batch_id": testset.obs["Site"].values,
            "error": test_errors.detach().cpu().numpy(),
            "model_id": batches_left_out[i],
        }
    )

    temp_df.to_csv(
        "../results/revision/analysis/batch_integration/"
        + data_name
        + "_"
        + batches_left_out[i]
        + "_prediction_errors_default.csv"
    )
    model = None
    test_predictions = None
    test_errors = None
    break
print("saved dgd prediction errors")


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
    # get test errors
    print("   computing test errors")
    test_errors = model.get_prediction_errors(
        test_predictions, model.test_set, reduction="sample"
    )

    ###
    # collect relevant errors and save in meaningfully plottable dataframe
    ###
    # make a dataframe per model with the following columns:
    # - sample id
    # - batch id of sample
    # - error of sample
    # - model id (in terms of batch left out)
    print("   collecting results")
    temp_df = pd.DataFrame(
        {
            "sample_id": testset.obs.index,
            "batch_id": testset.obs["Site"].values,
            "error": test_errors.detach().cpu().numpy(),
            "model_id": batches_left_out[i],
        }
    )

    temp_df.to_csv(
        "../results/revision/analysis/batch_integration/"
        + data_name
        + "_"
        + batches_left_out[i]
        + "_prediction_errors_supervised.csv"
    )
    model = None
    test_predictions = None
    test_errors = None
print("saved dgd prediction errors")
#"""

# now do the three models with 75% of the data
random_seeds = [0, 37, 8970]
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
    # get test errors
    print("   computing test errors")
    test_errors = model.get_prediction_errors(
        test_predictions, model.test_set, reduction="sample"
    )

    ###
    # collect relevant errors and save in meaningfully plottable dataframe
    ###
    # make a dataframe per model with the following columns:
    # - sample id
    # - batch id of sample
    # - error of sample
    # - model id (in terms of batch left out)
    print("   collecting results")
    temp_df = pd.DataFrame(
        {
            "sample_id": testset.obs.index,
            "batch_id": testset.obs["Site"].values,
            "error": test_errors.detach().cpu().numpy(),
            "model_id": "75percent_rs" + str(rs),
        }
    )

    temp_df.to_csv(
        "../results/revision/analysis/batch_integration/"
        + data_name
        + "_75percent_rs" + str(rs)
        + "_prediction_errors.csv"
    )
    model = None
    test_predictions = None
    test_errors = None
    break
print("saved dgd prediction errors")