import scvi
import torch
import pandas as pd
import anndata as ad
import numpy as np
from omicsdgd import DGD
from omicsdgd.functions._data_manipulation import load_testdata_as_anndata

save_dir = "results/trained_models/"
data_name = "human_bonemarrow"
# in this dictionary we will have the fractions and the corresponding number of samples in the train set
fraction_dict = {0.01: 567, 0.1: 5671, 0.25: 14178, 0.5: 28357, 0.75: 42535, 1.0: 56714}

"""
Go through datasets and chosen models and compute reconstruction performances
"""
data_name = "human_bonemarrow"
adata = ad.read_h5ad("data/" + data_name + ".h5ad")
train_indides = np.where(adata.obs["train_val_test"] == "train")[0]
trainset = adata[train_indides, :].copy()
test_indides = np.where(adata.obs["train_val_test"] == "test")[0]
testset = adata[test_indides, :].copy()
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
trainset_all = trainset.copy()
df_subset_ids = pd.read_csv("data/" + data_name + "/data_subsets.csv")

for count, fraction in enumerate(fraction_dict.keys()):
    subset = fraction_dict[fraction]
    if fraction == 1.0:
        n_samples = len(trainset_all)
    else:
        train_indices = list(
            df_subset_ids[
                (df_subset_ids["fraction"] == fraction)
                & (df_subset_ids["include"] == 1)
            ]["sample_idx"].values
        )
        trainset = adata[train_indices].copy()
        n_samples = len(train_indices)
    print("loaded data")

    for random_seed in [0, 37, 8790]:
        print("###")
        print(random_seed)
        print("###")
        model_name = (
            "human_bonemarrow_l20_h2-3_rs" + str(random_seed) + "_subset" + str(subset)
        )
        if fraction == 1.0:
            if random_seed == 0:
                model_name = "human_bonemarrow_l20_h2-3_test50e"
            else:
                model_name = "human_bonemarrow_l20_h2-3_rs" + str(random_seed)

        if fraction == 1.0:
            model = DGD.load(
                data=trainset_all,
                save_dir=save_dir + data_name + "/",
                model_name=model_name,
            )
        else:
            model = DGD.load(
                data=trainset,
                save_dir=save_dir + data_name + "/",
                model_name=model_name,
            )
        model.init_test_set(testset)
        predictions = model.predict_from_representation(
            model.test_rep, model.correction_test_rep
        )
        loss = model.get_prediction_errors(predictions, model.test_set)
        print(loss.item())
        metrics_temp = pd.DataFrame(
            {
                "loss": [loss.item()],
                "n_samples": [n_samples],
                "fraction": [fraction],
                "model": ["multiDGD"],
                "random_seed": [random_seed],
            }
        )
        model = None
        if (count == 0) & (random_seed == 0):
            metrics_df = metrics_temp
        else:
            metrics_df = metrics_df.append(metrics_temp)

metrics_df.to_csv(
    "results/analysis/performance_evaluation/" + data_name + "_data_efficiency.csv"
)
