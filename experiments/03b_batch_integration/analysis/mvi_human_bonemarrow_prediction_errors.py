"""
This script analyses the effect on prediction and data integration
of leaving a batch out of training
"""

# imports
import pandas as pd
import numpy as np
import scvi
import torch
import anndata as ad
from omicsdgd.functions._analysis import compute_expression_error, balanced_accuracies

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
trainset.var_names_make_unique()
trainset.obs["modality"] = "paired"
scvi.model.MULTIVI.setup_anndata(trainset, batch_key="Site")
testset.var_names_make_unique()
testset.obs["modality"] = "paired"
scvi.model.MULTIVI.setup_anndata(testset, batch_key="Site")
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

# loop over models, make predictions and compute errors per test sample
batches_left_out = ["none", "site1", "site2", "site3", "site4"]
model_names = [
    "l20_e2_d2",
    "l20_e2_d2_leftout_site1_scarches",
    "l20_e2_d2_leftout_site2_scarches",
    "l20_e2_d2_leftout_site3_scarches",
    "l20_e2_d2_leftout_site4_scarches",
]

for i, model_name in enumerate(model_names):
    print(batches_left_out[i])
    if batches_left_out[i] != "none":
        batches = trainset.obs["Site"].unique()
        train_indices = [
            x
            for x in np.arange(len(trainset))
            if trainset.obs["Site"].values[x] != batches_left_out[i]
        ]
        model = scvi.model.MULTIVI.load(
            save_dir + "multiVI/" + data_name + "/" + model_name,
            adata=trainset[train_indices],
        )
    else:
        model = scvi.model.MULTIVI.load(
            save_dir + "multiVI/" + data_name + "/" + model_name, adata=trainset
        )

    expression_error = compute_expression_error(
        testset,
        model,
        library[:, 0].unsqueeze(1),
        modality_switch,
        error_type="mae_sample",
        batch_size=1000,
    )
    accessibility_error = balanced_accuracies(
        testset,
        model,
        library[:, 1].unsqueeze(1),
        modality_switch,
        0.5,
        batch_size=1000,
    )
    norm_expression_error = expression_error / expression_error.max()
    norm_accessibility_error = accessibility_error / accessibility_error.max()
    test_errors = norm_accessibility_error + norm_expression_error

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
            "error": test_errors,
            "model_id": batches_left_out[i],
        }
    )

    temp_df.to_csv(
        "../results/analysis/batch_integration/mvi_"
        + data_name
        + "_"
        + batches_left_out[i]
        + "_prediction_errors.csv"
    )
    model = None
    test_predictions = None
    test_errors = None
print("saved mvi prediction errors")