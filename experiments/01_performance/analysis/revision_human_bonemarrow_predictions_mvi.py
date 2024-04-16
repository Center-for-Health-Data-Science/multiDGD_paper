"""
This script analyses the effect on prediction and data integration
of leaving a batch out of training
"""

# imports
import os
import pandas as pd
import numpy as np
import anndata as ad
import torch
import scvi
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000

from omicsdgd import DGD
from omicsdgd.functions._analysis import compute_expression_error, binary_output_scores, auprc

####################
# collect test errors per model and sample
####################
# load data
print("loading data")
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

print("prepping for MultiVI")
trainset.var_names_make_unique()
trainset.obs["modality"] = "paired"
scvi.model.MULTIVI.setup_anndata(trainset, batch_key="Site")
testset.var_names_make_unique()
testset.obs["modality"] = "paired"
scvi.model.MULTIVI.setup_anndata(testset, batch_key="Site")

# make sure the results directory exists
result_path = "../results/revision/analysis/batch_integration/" + data_name + "/"
plot_path = "../results/revision/plots/" + data_name + "/"
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)


# loop over models, make predictions and compute errors per test sample
model_names = ["l20_e2_d2", "l20_e2_d2_rs37", "l20_e2_d2_rs8790"]
model_dir = "../results/trained_models/"
random_seeds = [0, 37, 8790]

for i, model_name in enumerate(model_names):
    print("loading model ", model_name)
    model = scvi.model.MULTIVI.load(
        model_dir + "multiVI/" + data_name + "/" + model_name, adata=trainset
    )
    trainset = None

    """
    print("   calculating rmse and ba per sample")
    rmse_sample = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_sample', reduction="sample")

    tpr_s, tnr_s, ba_s, _, _ = binary_output_scores(testset, model, library[:,1].unsqueeze(1), modality_switch, threshold=0.5, batch_size=128, reduction="sample", return_all=True)
    #auprc_s, auprc_f = auprc(testset, model, library[:,1].unsqueeze(1), modality_switch, batch_size=128)
    df_sample = pd.DataFrame(
        {
            "rmse": rmse_sample.detach().cpu().numpy(),
            "tpr": tpr_s.detach().cpu().numpy(),
            "tnr": tnr_s.detach().cpu().numpy(),
            "ba": ba_s.detach().cpu().numpy(),
            #"auprc": auprc_s.detach().cpu().numpy(),
            "batch": testset.obs["Site"].values,
        }
    )
    df_sample.to_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_RMSE-BA_samplewise_mvi.csv"
    )
    """
    print("   calculating rmse and ba per feature")
    rmse_feature = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_feature', reduction="feature")
    df_feature = pd.DataFrame(
        {
            "rmse": rmse_feature.detach().cpu().numpy(),
            "feature_name": testset.var_names[:modality_switch],
            "feature": "rna"
        }
    )
    tpr_f, tnr_f, ba_f, _, _ = binary_output_scores(testset, model, library[:,1].unsqueeze(1), modality_switch, threshold=0.5, batch_size=128, reduction="feature", return_all=True)
    df_feature = pd.concat(
        [
            df_feature,
            pd.DataFrame(
                {
                    "tpr": tpr_f.detach().cpu().numpy(),
                    "tnr": tnr_f.detach().cpu().numpy(),
                    "ba": ba_f.detach().cpu().numpy(),
                    #"auprc": auprc_f.detach().cpu().numpy(),
                    "feature_name": testset.var_names[modality_switch:],
                    "feature": "atac"
                }
            )
        ]
    )
    df_feature.to_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_RMSE-BA_genewise_mvi.csv"
    )
    """
    print("   calculating rmse and ba per feature")
    test_errors = model.get_prediction_errors(
        test_predictions, model.test_set, reduction="gene"
    )
    #test_predictions = None
    df_sample, df_gene = calculate_mean_errors(test_errors, testset, modality_switch)
    df_sample.to_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_errors_samplewise_mvi.csv"
    )
    df_gene.to_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_errors_genewise_mvi.csv"
    )
    test_errors = model.get_prediction_errors(
        test_predictions, model.test_set, reduction="sample"
    )
    temp_df = pd.DataFrame(
        {
            "sample_id": testset.obs.index,
            "batch_id": testset.obs["Site"].values,
            "error": test_errors.detach().cpu().numpy(),
        }
    )
    """
    model = None
    test_predictions = None
    test_errors = None
    rmse_sample, rmse_feature = None, None
    tpr_s, tpr_f, tnr_s, tnr_f, ba_s, ba_f = None, None, None, None, None, None
    break
print("saved mvi predictions")