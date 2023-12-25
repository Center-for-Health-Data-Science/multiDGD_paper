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
from omicsdgd.functions._metrics import clustering_metric
from sklearn.metrics import silhouette_score

####################
# collect test errors per model and sample
####################
# load data
save_dir = "../results/trained_models/"
data_name = "human_bonemarrow"
adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
adata.X = adata.layers["counts"] # I seem to have to do it again
modality_switch = 13431

# train-validation-test split for reproducibility
# best provided as list [[train_indices], [validation_indices]]
train_val_split = [
    list(np.where(adata.obs["train_val_test"] == "train")[0]),
    list(np.where(adata.obs["train_val_test"] == "validation")[0]),
]

valset = adata[adata.obs["train_val_test"] == "validation"].copy()
valset.obs["modality"] = "paired"
testset = adata[adata.obs["train_val_test"] == "test"].copy()
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

###
# prep unpaired data if not already existing
###
df_unpaired = pd.read_csv('../../data/'+data_name+'_unpairing.csv')

# make sure the results directory exists
import os
result_path = "../results/revision/analysis/mosaic/" + data_name + "/"
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


# loop over models, make predictions and compute errors per test sample
model_names = [
    "human_bonemarrow_l20_h2-3_test50e",
    "human_bonemarrow_l20_h2-3_rs0_mosaic0.1percent_test50e",
    "human_bonemarrow_l20_h2-3_rs0_mosaic0.5percent_test50e",
    "human_bonemarrow_l20_h2-3_rs0_mosaic0.9percent_test50e"
]
fractions_unpaired = [0, 0.1, 0.5, 0.9]
for i, model_name in enumerate(model_names):
    print("loading model " + model_name)
    fraction_unpaired = fractions_unpaired[i]
    if fraction_unpaired > 0:
        mod_1_indices = df_unpaired[
            (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "rna")
        ]["sample_idx"].values
        mod_2_indices = df_unpaired[
            (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "atac")
        ]["sample_idx"].values
        remaining_indices = df_unpaired[
            (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "paired")
        ]["sample_idx"].values

        var_before = adata.var.copy()
        if fraction_unpaired < 1.0:
            adata_rna = adata[mod_1_indices, adata.var["feature_types"] == "GEX"].copy()
            adata_rna.obs["modality"] = "GEX"
            adata_atac = adata[mod_2_indices, adata.var["feature_types"] == "ATAC"].copy()
            adata_atac.obs["modality"] = "ATAC"
            adata_multi = adata[remaining_indices, :].copy()
            adata_multi.obs["modality"] = "paired"
            adata_unpaired = ad.concat([adata_multi, adata_rna, adata_atac], join="outer")
            adata_rna, adata_atac, adata_multi = None, None, None
        else:
            adata_unpaired = adata[mod_1_indices,:].copy()
            adata_unpaired.obs['modality'] = 'GEX'
            adata_temp = adata[mod_2_indices,:].copy()
            adata_temp.obs['modality'] = 'ATAC'
            #adata_unpaired = adata_unpaired.concatenate(adata_temp)
            adata_unpaired = ad.concat([adata_unpaired, adata_temp], join="outer")
            adata_temp = None

        #adata = adata_unpaired.concatenate(valset)
        adata_new = ad.concat([adata_unpaired, valset], join="inner")
        adata_new.var = var_before

        # update the train-val split
        train_val_split = [
            list(np.where(adata_new.obs["train_val_test"] == "train")[0]),
            list(np.where(adata_new.obs["train_val_test"] == "validation")[0]),
        ]
        model = DGD.load(
            data=adata_new[train_val_split[0]], save_dir=save_dir + data_name + "/", model_name=model_name
        )
    else:
        model = DGD.load(
            data=adata[train_val_split[0]], save_dir=save_dir + data_name + "/", model_name=model_name
        )
    
    # get clustering metrics
    if fraction_unpaired > 0.0:
        asw = silhouette_score(model.representation.z.detach().cpu(), adata_new[train_val_split[0]].obs["Site"].values)
        asw_modality = silhouette_score(model.representation.z.detach().cpu(), adata_new[train_val_split[0]].obs["modality"].values)
        ari = clustering_metric(model.representation, model.gmm, adata_new[train_val_split[0]].obs["cell_type"].values)
        adata_new = None
    else:
        asw = silhouette_score(model.representation.z.detach().cpu(), adata[train_val_split[0]].obs["Site"].values)
        asw_modality = None
        ari = clustering_metric(model.representation, model.gmm, adata[train_val_split[0]].obs["cell_type"].values)
    clustering_df = pd.DataFrame({"fraction_unpaired": fraction_unpaired, "silhouette": asw, "silhouette (modality)": asw_modality, "ARI (GMM)": ari, "model_name": model_name}, index=[0])
    # save the dataframe
    clustering_df.to_csv(
        result_path
        + data_name
        + "_rs0_mosaic"
        + str(fraction_unpaired)
        + "_clustering_metrics.csv"
    )

    model.init_test_set(testset)

    # get test predictions
    print("   predicting test samples")
    test_predictions = model.predict_from_representation(
        model.test_rep, model.correction_test_rep
    )
    rmse_sample = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_sample', reduction="sample")
    metrics_temp = testset_reconstruction_evaluation_extended(
        testset, model, modality_switch, library, thresholds=[0.2]
    )
    # save
    metrics_temp.to_csv(
        result_path
        + data_name
        + "_rs0_mosaic"
        + str(fraction_unpaired)
        + "_recon_metrics.csv"
    )
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
        + "_rs0_mosaic"
        + str(fraction_unpaired)
        + "_RMSE-BA_samplewise.csv"
    )
    rmse_feature = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_feature', reduction="feature")
    df_feature = pd.DataFrame(
        {
            "rmse": rmse_feature.detach().cpu().numpy(),
            "feature_name": testset.var_names[:modality_switch],
            "feature": "rna"
        }
    )
    tpr_f, tnr_f, ba_f, _, _ = binary_output_scores(testset, model, library[:,1].unsqueeze(1), modality_switch, threshold=0.2, reduction="feature", return_all=True)
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
        + "_rs0_mosaic"
        + str(fraction_unpaired)
        + "_RMSE-BA_genewise.csv"
    )
    # save test predictions as numpy
    test_errors = model.get_prediction_errors(
        test_predictions, model.test_set, reduction="gene"
    )
    #test_predictions = None
    df_sample, df_gene = calculate_mean_errors(test_errors, testset, modality_switch)
    df_sample.to_csv(
        result_path
        + data_name
        + "_rs0_mosaic"
        + str(fraction_unpaired)
        + "_errors_samplewise.csv"
    )
    df_gene.to_csv(
        result_path
        + data_name
        + "_rs0_mosaic"
        + str(fraction_unpaired)
        + "_errors_genewise.csv"
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
    temp_df.to_csv(
        result_path
        + data_name
        + "_rs0_mosaic"
        + str(fraction_unpaired)
        + "_prediction_errors_supervised.csv"
    )
    test_predictions = None
    test_errors = None
    model = None
print("saved dgd predictions")