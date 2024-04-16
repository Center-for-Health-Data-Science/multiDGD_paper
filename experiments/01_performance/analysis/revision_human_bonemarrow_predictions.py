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
from omicsdgd.functions._analysis import testset_reconstruction_evaluation_extended, compute_expression_error, binary_output_scores, calculate_mean_errors, alternative_ATAC_metrics, compute_count_error, spearman_corr

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
    if i < 5:
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
    auprc, spearman = alternative_ATAC_metrics(model, testset, modality_switch, library[:,1].unsqueeze(1), axis=None)
    spearman_rna = spearman_corr(model, testset, modality_switch, library[:,0].unsqueeze(1), axis=None, mod_id=0)
    # save auprc and spearman
    pd.DataFrame(
        {
            "auprc": [auprc],
            "spearman": [spearman],
            "spearman_rna": [spearman_rna],
        }
    ).to_csv(
        result_path
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_"
        + model_types[i]
        + "_ncomp"
        + str(n_components[i])
        + "_auprc_spearman.csv"
    )
    #rmse_atac_sample = compute_count_error(testset, model, library[:,1].unsqueeze(1), modality_switch, error_type='rmse_sample', reduction="sample", mod_id=1)
    #rmse_atac_feature = compute_count_error(testset, model, library[:,1].unsqueeze(1), modality_switch, error_type='rmse_feature', reduction="feature", mod_id=1)

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
        tpr_s, tnr_s, ba_s, _, _ = binary_output_scores(testset, model, library[:,1].unsqueeze(1), modality_switch, threshold=0.2, batch_size=128, reduction="sample", return_all=True)
        auprc_s, spearman_s = alternative_ATAC_metrics(model, testset, modality_switch, library[:,1].unsqueeze(1), axis=0)
        df_sample = pd.DataFrame(
            {
                "rmse": rmse_sample.detach().cpu().numpy(),
                "tpr": tpr_s.detach().cpu().numpy(),
                "tnr": tnr_s.detach().cpu().numpy(),
                "ba": ba_s.detach().cpu().numpy(),
                "auprc": auprc_s,
                "spearman": spearman_s,
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
    # free memory
    rmse_sample, df_sample, tpr_s, tnr_s, ba_s, auprc_s, spearman_s = None, None, None, None, None, None, None
    # free up cuda memory
    torch.cuda.empty_cache()

    rmse_feature = compute_expression_error(testset, model, library[:,0].unsqueeze(1), modality_switch, error_type='rmse_feature', reduction="feature")
    df_feature = pd.DataFrame(
        {
            "rmse": rmse_feature.detach().cpu().numpy(),
            "feature_name": testset.var_names[:modality_switch],
            "feature": "rna"
        }
    )
    if model_types[i] != "scDGD":
        tpr_f, tnr_f, ba_f, _, _ = binary_output_scores(testset, model, library[:,1].unsqueeze(1), modality_switch, threshold=0.2, batch_size=128, reduction="feature", return_all=True)
        auprc_f, spearman_f = alternative_ATAC_metrics(model, testset, modality_switch, library[:,1].unsqueeze(1), axis=1)
        df_feature = pd.concat(
            [
                df_feature,
                pd.DataFrame(
                    {
                        "tpr": tpr_f.detach().cpu().numpy(),
                        "tnr": tnr_f.detach().cpu().numpy(),
                        "ba": ba_f.detach().cpu().numpy(),
                        "auprc": auprc_f,
                        "spearman": spearman_f,
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
    # free memory
    test_errors, test_predictions, temp_df = None, None, None
    rmse_feature, df_feature, tpr_f, tnr_f, ba_f, aucpr_f, spearman_f = None, None, None, None, None, None, None

    model = None
    test_predictions = None
    test_errors = None

    # free up cuda memory
    torch.cuda.empty_cache()
print("saved dgd predictions")