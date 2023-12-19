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


# loop over models, make predictions and compute errors per test sample
model_names = [
    "human_bonemarrow_l20_h2-3_test50e",
    "human_bonemarrow_l20_h2-3_rs37",
    "human_bonemarrow_l20_h2-3_rs8790",
    "human_bonemarrow_l20_h2-3_rs0_ncomp1_test50e",
    "human_bonemarrow_l20_h2-3_rs37_ncomp1_test50e",
    "human_bonemarrow_l20_h2-3_rs8790_ncomp1_test50e",
    "human_bonemarrow_l20_h2-3_rs0_noCovariate_test50e",
    "human_bonemarrow_l20_h2-3_rs37_noCovariate_test50e",
    "human_bonemarrow_l20_h2-3_rs8790_noCovariate_test50e",
    "human_bonemarrow_l20_h3_rs0_scDGD_test50e",
    "human_bonemarrow_l20_h3_rs37_scDGD_test50e",
    "human_bonemarrow_l20_h3_rs8790_scDGD_test50e"
]
random_seeds = [0, 37, 8790]*4
model_types = ["", "", "", "", "", "", "noCovariate", "noCovariate", "noCovariate", "scDGD", "scDGD", "scDGD"]
n_components = [22, 22, 22, 1, 1, 1, 22, 22, 22, 22, 22, 22]
for i, model_name in enumerate(model_names):
    if i < 6:
        continue
    print("loading model " + model_name)
    if model_types[i] == "scDGD":
        if random_seeds[i] == 0:
            modality_switch = 13431
            trainset = trainset[:, :modality_switch]
            testset = testset[:, :modality_switch]
    model = DGD.load(
        data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
    )
    model.init_test_set(testset)

    # get test predictions
    print("   predicting test samples")
    if model_types[i] not in ["scDGD", "noCovariate"]:
        test_predictions = model.predict_from_representation(
            model.test_rep, model.correction_test_rep
        )
    else:
        test_predictions = model.predict_from_representation(
            model.test_rep
        )
    rmse_sample = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_sample', reduction="sample")
    if model_types[i] != "scDGD":
        metrics_temp = testset_reconstruction_evaluation_extended(
            testset, model, modality_switch, library, thresholds=[0.2]
        )
        # save
        metrics_temp.to_csv(
            result_path
            + data_name
            + "_rs"
            + str(random_seeds[i])
            + "_"
            + model_types[i]
            + "_ncomp"
            + str(n_components[i])
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
    else:
        df_sample = pd.DataFrame(
            {
                "rmse": rmse_sample.detach().cpu().numpy(),
                "batch": testset.obs["Site"].values,
            }
        )
    df_sample.to_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_"
        + model_types[i]
        + "_ncomp"
        + str(n_components[i])
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
    if model_types[i] != "scDGD":
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
        + "_rs"
        + str(random_seeds[i])
        + "_"
        + model_types[i]
        + "_ncomp"
        + str(n_components[i])
        + "_RMSE-BA_genewise.csv"
    )
    # save test predictions as numpy
    if model_types[i] != "scDGD":
        print("   computing test errors")
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
            + "_"
            + model_types[i]
            + "_ncomp"
            + str(n_components[i])
            + "_errors_samplewise.csv"
        )
        df_gene.to_csv(
            result_path
            + data_name
            + "_rs"
            + str(random_seeds[i])
            + "_"
            + model_types[i]
            + "_ncomp"
            + str(n_components[i])
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
        + "_rs"
        + str(random_seeds[i])
        + "_"
        + model_types[i]
        + "_ncomp"
        + str(n_components[i])
        + "_prediction_errors_supervised.csv"
    )
    model = None
    test_predictions = None
    test_errors = None
print("saved dgd predictions")