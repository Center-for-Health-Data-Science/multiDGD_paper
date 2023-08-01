import scvi
import torch
import numpy as np
import pandas as pd
import anndata as ad

from omicsdgd.functions._data_manipulation import load_testdata_as_anndata

save_dir = "results/trained_models/"
data_name = "human_bonemarrow"

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

subset_samples = [567, 5671, 14178, 28357, 42535, 56714]
subset_samples.reverse()
fraction_options = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
fraction_options.reverse()

for count, fraction in enumerate(fraction_options):
    print("###")
    print(fraction)
    print("###")
    subset = subset_samples[count]
    train_indices = list(
        df_subset_ids[
            (df_subset_ids["fraction"] == fraction) & (df_subset_ids["include"] == 1)
        ]["sample_idx"].values
    )
    if fraction == 1.0:
        # trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
        n_samples = len(trainset_all)
        trainset_all.var_names_make_unique()
        trainset_all.obs["modality"] = "paired"
        scvi.model.MULTIVI.setup_anndata(trainset_all, batch_key="Site")
    else:
        trainset = adata[train_indices].copy()
        n_samples = len(train_indices)
        trainset.var_names_make_unique()
        trainset.obs["modality"] = "paired"
        scvi.model.MULTIVI.setup_anndata(trainset, batch_key="Site")
    testset.var_names_make_unique()
    testset.obs["modality"] = "paired"
    scvi.model.MULTIVI.setup_anndata(testset, batch_key="Site")
    print("loaded data")

    for random_seed in [0, 37, 8790]:
        model_name = "l20_e2_d2_rs" + str(random_seed) + "_subset" + str(subset)
        if fraction == 1.0:
            if random_seed == 0:
                model_name = "l20_e2_d2"
            else:
                model_name = "l20_e2_d2_rs" + str(random_seed)

        if fraction == 1.0:
            model = scvi.model.MULTIVI.load(
                save_dir + "multiVI/" + data_name + "/" + model_name, adata=trainset_all
            )
        else:
            model = scvi.model.MULTIVI.load(
                save_dir + "multiVI/" + data_name + "/" + model_name, adata=trainset
            )
        elbo = model.get_elbo(testset)
        metrics_temp = pd.DataFrame(
            {
                "loss": [elbo.item()],
                "n_samples": [n_samples],
                "fraction": [fraction],
                "model": ["multiVI"],
                "random_seed": [random_seed],
            }
        )
        model = None
        if (count == 0) & (random_seed == 0):
            metrics_mvi = metrics_temp
        else:
            metrics_mvi = metrics_mvi.append(metrics_temp)

metrics_mvi.to_csv(
    "results/analysis/performance_evaluation/" + data_name + "_data_efficiency_mvi.csv"
)

print("done")
