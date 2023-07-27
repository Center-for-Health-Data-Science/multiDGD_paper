import pandas as pd
import anndata as ad
from omicsdgd import DGD
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
#import scvi
import torch
from omicsdgd.functions._data_manipulation import load_testdata_as_anndata
from omicsdgd.functions._analysis import make_palette_from_meta

save_dir = "results/trained_models/"
data_name = "human_bonemarrow"
model_name = "human_bonemarrow_l20_h2-3_rs0_unpaired0percent"
#model_name = "human_bonemarrow_l20_h2-3_test10e"
#model_name = "human_bonemarrow_l20_h2-3_test50e"
fraction_unpaired = 0.
# fraction_unpaired_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

####################
# prepare data and load model
####################
is_train_df = pd.read_csv("data/" + data_name + "/train_val_test_split.csv")
train_indices = is_train_df[is_train_df["is_train"] == "train"]["num_idx"].values
df_unpaired = pd.read_csv("data/" + data_name + "/unpairing.csv")
adata = ad.read_h5ad("data/" + data_name + "/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")
adata.X = adata.layers["counts"]
modality_switch = np.where(adata.var["feature_types"] == "ATAC")[0][0]
print(modality_switch)
# make test set now to be able to free memory earlier
adata_test = adata[is_train_df[is_train_df["is_train"] == "iid_holdout"]["num_idx"].values, :].copy()
print("loaded data")

if fraction_unpaired == 0.0:
    model = DGD.load(data=adata[train_indices, :], save_dir=save_dir + data_name + "/", model_name=model_name)
    adata = None
else:
    mod_1_indices = df_unpaired[
        (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "rna")
    ]["sample_idx"].values
    mod_2_indices = df_unpaired[
        (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "atac")
    ]["sample_idx"].values
    if fraction_unpaired < 1.:
        # train-validation-test split for reproducibility
        remaining_indices = df_unpaired[
            (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "paired")
        ]["sample_idx"].values
        print("made indices")
        adata_rna = adata[mod_1_indices, adata.var["feature_types"] == "GEX"].copy()
        print("copied rna")
        adata_atac = adata[mod_2_indices, adata.var["feature_types"] == "ATAC"].copy()
        print("copied atac")
        adata_multi = adata[remaining_indices, :].copy()
        print("copied rest")
        adata_unpaired = scvi.data.organize_multiome_anndatas(adata_multi, adata_rna, adata_atac)
        print("organized data")
        adata_rna, adata_atac, adata_multi = None, None, None

        model = DGD.load(data=adata_unpaired, save_dir=save_dir + data_name + "/", model_name=model_name)
        adata_unpaired = None
    else:
        adata_unpaired = adata[mod_1_indices,:].copy()
        adata_unpaired.obs['modality'] = 'GEX'
        adata_temp = adata[mod_2_indices,:].copy()
        adata_temp.obs['modality'] = 'ATAC'
        adata_unpaired = adata_unpaired.concatenate(adata_temp)
        print("organized data")
        adata, adata_temp = None, None
        model = DGD.load(data=adata_unpaired, save_dir=save_dir + data_name + "/", model_name=model_name)
        adata_unpaired = None
print("loaded model")


####################
# get predictions and save
####################

def binarize(x, threshold=0.5):
    x[x >= threshold] = 1
    x[x < threshold] = 0
    return x

def get_rmse(preds, targets, scalings):
    '''returns sample-wise and overall RMSE
    preds: torch tensor of predictions
    targets: torch tensor of targets
    scalings: torch tensor of scaling factors'''
    print(preds.shape, targets.shape, scalings.shape)
    preds *= scalings
    error = targets - preds
    mse_per_sample = torch.mean(error**2, axis=1)
    rmse_per_sample = torch.sqrt(mse_per_sample)
    rmse = torch.sqrt(mse_per_sample.mean()).item()
    return rmse_per_sample, rmse

def get_balanced_accuracy(preds, targets, scalings):
    '''returns sample-wise and overall balanced accuracy
    preds: torch tensor of predictions
    targets: torch tensor of targets
    scalings: torch tensor of scaling factors'''
    print(preds.shape, targets.shape, scalings.shape)
    preds *= scalings
    bin_x = binarize(targets, threshold=0.2)
    bin_y = binarize(preds, threshold=0.2)
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
n_test_samples = len(is_train_df[is_train_df["is_train"] == "iid_holdout"])
print("number of test samples and shape of learned representation")
print(n_test_samples)
print(model.test_rep.z.shape)
#predictions = model.predict_from_representation(model.test_rep, model.correction_test_rep)
predictions = model.decoder_forward(model.test_rep.z.shape[0], np.arange(model.test_rep.z.shape[0]))
print("predictions")
print(predictions[0].shape, predictions[1].shape)
model = None

# get data sets and scaling factors
rna_scaling_factors = torch.Tensor(np.asarray(adata_test.X[:, :modality_switch].sum(axis=1)))
atac_scaling_factors = torch.Tensor(np.asarray(adata_test.X[:, modality_switch:].sum(axis=1)))

print("multi reconstructions")
# rna
rmse_per_sample_multi, rmse_multi = get_rmse(
    predictions[0][:n_test_samples, :].detach().clone(),
    torch.Tensor(adata_test.X[:, :modality_switch].copy().todense()),
    rna_scaling_factors)
print("RMSE: ", rmse_multi)

# atac
balanced_accuracy_per_sample_multi, balanced_accuracy_multi = get_balanced_accuracy(
    predictions[1][:n_test_samples, :].detach().clone(),
    torch.Tensor(adata_test.X[:, modality_switch:].copy().todense()),
    atac_scaling_factors)
print("balanced accuracy: ", balanced_accuracy_multi)

print("doing RNA")
# compute RNA prediction errors from test set with ATAC data only
print((2 * n_test_samples))
rmse_per_sample_atac, rmse_atac = get_rmse(
    predictions[0][(2 * n_test_samples):, :].detach().clone(),
    torch.Tensor(adata_test.X[:, :modality_switch].copy().todense()),
    rna_scaling_factors)
print("RMSE: ", rmse_atac)

print("doing ATAC")
# compute ATAC prediction errors from test set with RNA data only
balanced_accuracy_per_sample_rna, balanced_accuracy_rna = get_balanced_accuracy(
    predictions[1][n_test_samples:(2 * n_test_samples), :].detach().clone(),
    torch.Tensor(adata_test.X[:, modality_switch:].copy().todense()),
    atac_scaling_factors)
print("balanced accuracy: ", balanced_accuracy_rna)

df_metric = pd.DataFrame(
    {
        "fraction": [fraction_unpaired] * 4,
        "value": [rmse_multi, balanced_accuracy_multi, rmse_atac, balanced_accuracy_rna],
        "metric": ["RMSE", "balanced_accuracy", "RMSE", "balanced_accuracy"],
        "modality": ["paired", "paired", "ATAC", "RNA"],
    }
)
df_metric.to_csv("results/analysis/modality_integration/" + model_name + "_predictive_performance.csv", index=False)

if fraction_unpaired == 0:
    df_out = pd.DataFrame({
        "reconstruction": rmse_per_sample_multi.numpy(),
        "prediction": rmse_per_sample_atac.numpy(),
        "modality": "RNA",
        "metric": "RMSE"
    })
    df_out = pd.concat(
        [
            df_out,
            pd.DataFrame({
                "reconstruction": balanced_accuracy_per_sample_multi.numpy(),
                "prediction": balanced_accuracy_per_sample_rna.numpy(),
                "modality": "ATAC",
                "metric": "balanced accuracy"
            })
        ]
    )
    df_out.to_csv(
        "results/analysis/modality_integration/" + model_name + "_prediction_errors_samplewise.csv", index=False
    )
