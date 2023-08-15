import pandas as pd
import anndata as ad
from omicsdgd import DGD
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scvi
import torch
from omicsdgd.functions._data_manipulation import load_testdata_as_anndata
from omicsdgd.functions._analysis import make_palette_from_meta

save_dir = "../results/trained_models/"
data_name = "human_bonemarrow"
fraction_unpaired = 0.
model_name = "l20_e2_d2"

####################
# prepare data and load model
####################
adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
adata.X = adata.layers["counts"]
adata.obs["modality"] = "paired"
adata.var_names_make_unique()
train_indices = np.where(adata.obs["train_val_test"] == "train")[0]
trainset = adata[train_indices, :].copy()
test_indices = np.where(adata.obs["train_val_test"] == "test")[0]
adata_test = adata[test_indices, :].copy()
modality_switch = np.where(adata.var["feature_types"] == "ATAC")[0][0]

scvi.model.MULTIVI.setup_anndata(trainset, batch_key="Site")

#adata_rna = adata_test[:, adata_test.var["feature_types"] == "GEX"].copy()
#adata_rna.obs["modality"] = "GEX"
#adata_atac = adata_test[:, adata_test.var["feature_types"] == "ATAC"].copy()
#adata_atac.obs["modality"] = "ATAC"
adata = None
#adata_unpaired_test = scvi.data.organize_multiome_anndatas(
#    adata_test, adata_rna, adata_atac
#)
#print("   organized data")
#adata_rna, adata_atac, adata_test = None, None, None
#scvi.model.MULTIVI.setup_anndata(adata_unpaired_test, batch_key="Site")
scvi.model.MULTIVI.setup_anndata(adata_test, batch_key="Site")

print("loaded data")

model = scvi.model.MULTIVI.load(save_dir + "multiVI/" + data_name + "/" + model_name, adata=trainset)

print("loaded model")


####################
# get predictions and save
####################


def binarize(x, threshold=0.5):
    x[x >= threshold] = 1
    x[x < threshold] = 0
    return x


def get_rmse(preds, targets, scalings):
    """returns sample-wise and overall RMSE
    preds: torch tensor of predictions
    targets: torch tensor of targets
    scalings: torch tensor of scaling factors"""
    print(preds.shape, targets.shape, scalings.shape)
    preds *= scalings
    error = targets - preds
    mse_per_sample = torch.mean(error**2, axis=1)
    rmse_per_sample = torch.sqrt(mse_per_sample)
    rmse = torch.sqrt(mse_per_sample.mean()).item()
    return rmse_per_sample, rmse


def get_balanced_accuracy(preds, targets, scalings):
    """returns sample-wise and overall balanced accuracy
    preds: torch tensor of predictions
    targets: torch tensor of targets
    scalings: torch tensor of scaling factors"""
    print(preds.shape, targets.shape, scalings.shape)
    preds *= scalings
    bin_x = binarize(targets, threshold=0.5)
    bin_y = binarize(preds, threshold=0.5)
    p = bin_x == 1
    pp = bin_y == 1
    true_positives = torch.logical_and(p, pp).sum(-1)
    true_negatives = torch.logical_and(~p, ~pp).sum(-1)
    false_positives = (bin_y > bin_x).sum(-1)
    false_negatives = (bin_y < bin_x).sum(-1)
    # first sample-wise
    tpr = true_positives / (true_positives + false_negatives)  # sensitivity
    tnr = true_negatives / (true_negatives + false_positives)  # specificity
    balanced_accuracy_per_sample = (tpr + tnr) / 2
    tpr = true_positives.sum() / (true_positives.sum() + false_negatives.sum())  # sensitivity
    tnr = true_negatives.sum() / (true_negatives.sum() + false_positives.sum())  # specificity
    balanced_accuracy = ((tpr + tnr) / 2).item()
    return balanced_accuracy_per_sample, balanced_accuracy


# get predictions
n_test_samples = len(test_indices)

# predictions = model.predict_from_representation(model.test_rep, model.correction_test_rep)
predictions_rna = model.get_normalized_expression(adata_test).values
predictions_atac = model.get_accessibility_estimates(adata_test).values

# get data sets and scaling factors
rna_scaling_factors = torch.Tensor(np.asarray(adata_test.X[:, :modality_switch].sum(axis=1)))
atac_scaling_factors = torch.Tensor(np.asarray(adata_test.X[:, modality_switch:].sum(axis=1)))

print("multi reconstructions")
# rna
rmse_per_sample_multi, rmse_multi = get_rmse(
    torch.Tensor(predictions_rna), torch.Tensor(adata_test.X[:, :modality_switch].copy().todense()), rna_scaling_factors
)
print("RMSE: ", rmse_multi)

# atac
balanced_accuracy_per_sample_multi, balanced_accuracy_multi = get_balanced_accuracy(
    torch.Tensor(predictions_atac),
    torch.Tensor(adata_test.X[:, modality_switch:].copy().todense()),
    atac_scaling_factors,
)
print("balanced accuracy: ", balanced_accuracy_multi)

print("doing RNA")
adata_atac = adata_test[:, adata_test.var["feature_types"] == "ATAC"].copy()
adata_atac.obs["modality"] = "ATAC"
scvi.model.MULTIVI.prepare_query_anndata(adata_atac, model)
print(adata_atac)
print(adata_atac.X[:,:modality_switch].sum())
print(adata_atac.obs["modality"])
predictions_rna = model.get_normalized_expression(adata_atac).values
# compute RNA prediction errors from test set with ATAC data only
print((2 * n_test_samples))
rmse_per_sample_atac, rmse_atac = get_rmse(
    torch.Tensor(predictions_rna), torch.Tensor(adata_test.X[:, :modality_switch].copy().todense()), rna_scaling_factors
)
print("RMSE: ", rmse_atac)

print("doing ATAC")
adata_rna = adata_test[:, adata_test.var["feature_types"] == "GEX"].copy()
adata_rna.obs["modality"] = "GEX"
scvi.model.MULTIVI.prepare_query_anndata(adata_rna, model)
print(adata_rna)
print(adata_rna.X[:,modality_switch:].sum())
print(adata_rna.obs["modality"])
predictions_atac = model.get_accessibility_estimates(adata_rna).values
# compute ATAC prediction errors from test set with RNA data only
balanced_accuracy_per_sample_rna, balanced_accuracy_rna = get_balanced_accuracy(
    torch.Tensor(predictions_atac),
    torch.Tensor(adata_test.X[:, modality_switch:].copy().todense()),
    atac_scaling_factors,
)
print("balanced accuracy: ", balanced_accuracy_rna)

df_metric = pd.DataFrame(
    {
        "fraction": [fraction_unpaired] * 4,
        "value": [rmse_multi, balanced_accuracy_multi, rmse_atac, balanced_accuracy_rna],
        "metric": ["RMSE", "balanced_accuracy", "RMSE", "balanced_accuracy"],
        "modality": ["paired", "paired", "ATAC", "RNA"],
    }
)
df_metric.to_csv("../results/analysis/modality_integration/mvi_" + model_name + "_predictive_performance.csv", index=False)

if fraction_unpaired == 0:
    df_out = pd.DataFrame(
        {
            "reconstruction": rmse_per_sample_multi.numpy(),
            "prediction": rmse_per_sample_atac.numpy(),
            "modality": "RNA",
            "metric": "RMSE",
        }
    )
    df_out = pd.concat(
        [
            df_out,
            pd.DataFrame(
                {
                    "reconstruction": balanced_accuracy_per_sample_multi.numpy(),
                    "prediction": balanced_accuracy_per_sample_rna.numpy(),
                    "modality": "ATAC",
                    "metric": "balanced accuracy",
                }
            ),
        ]
    )
    df_out.to_csv(
        "../results/analysis/modality_integration/mvi_" + model_name + "_prediction_errors_samplewise.csv", index=False
    )
