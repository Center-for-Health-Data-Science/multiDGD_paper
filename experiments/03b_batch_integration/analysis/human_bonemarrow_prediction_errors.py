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
save_dir = "results/trained_models/"
data_name = "human_bonemarrow"
adata = ad.read_h5ad("data/" + data_name + ".h5ad")
train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
test_indices = list(np.where(adata.obs["train_val_test"] == "test")[0])
trainset = adata[train_indices, :].copy()
testset = adata[test_indices, :].copy()
batches = trainset.obs["Site"].unique()

# loop over models, make predictions and compute errors per test sample
batches_left_out = ["none", "site1", "site2", "site3", "site4"]
model_names = [
    "human_bonemarrow_l20_h2-3_test10e",
    "human_bonemarrow_l20_h2-3_leftout_site1",
    "human_bonemarrow_l20_h2-3_leftout_site2",
    "human_bonemarrow_l20_h2-3_leftout_site3",
    "human_bonemarrow_l20_h2-3_leftout_site4",
]
for i, model_name in enumerate(model_names):
    print(batches_left_out[i])
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
    print("predicting test samples")
    test_predictions = model.predict_from_representation(
        model.test_rep, model.correction_test_rep
    )
    # get test errors
    print("computing test errors")
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
    print("collecting results")
    temp_df = pd.DataFrame(
        {
            "sample_id": testset.obs.index,
            "batch_id": testset.obs["Site"].values,
            "error": test_errors.detach().cpu().numpy(),
            "model_id": batches_left_out[i],
        }
    )
    # if i == 0:
    #    df = temp_df
    # else:
    #    temp_df = temp_df[temp_df['batch_id'] == batches_left_out[i]]
    #    print(temp_df)
    #    df = df.append(temp_df)

    temp_df.to_csv(
        "results/analysis/batch_integration/"
        + data_name
        + "_"
        + batches_left_out[i]
        + "_prediction_errors.csv"
    )
    model = None
    test_predictions = None
    test_errors = None
