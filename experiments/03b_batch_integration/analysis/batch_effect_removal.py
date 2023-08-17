"""
This script analyses the effect on prediction and data integration
of leaving a batch out of training
"""

# imports
import os
import pandas as pd
import numpy as np
import scvi
import anndata as ad
from omicsdgd import DGD
from sklearn.metrics import silhouette_score

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

# trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)

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

    rep = model.test_rep.z.detach().cpu().numpy()
    asw = silhouette_score(rep, testset.obs["Site"])
    print(asw)

    temp_df = pd.DataFrame(
        {"batch": [batches_left_out[i]], "ASW": [asw], "model": ["multiDGD"]}
    )
    if i == 0:
        df = temp_df
    else:
        df = pd.concat([df, temp_df])

    model = None
    test_predictions = None
    test_errors = None

if not os.path.exists("../results/analysis/batch_integration"):
    os.makedirs("../results/analysis/batch_integration")
df.to_csv("../results/analysis/batch_integration/" + data_name + "_batch_effect.csv")
print("saved dgd batch effects")


# now multiVI
print("doing multiVI now")
model_dir = "../results/trained_models/multiVI/" + data_name + "/"
model_names = [
    "l20_e2_d2",
    "l20_e2_d2_leftout_site1_scarches",
    "l20_e2_d2_leftout_site2_scarches",
    "l20_e2_d2_leftout_site3_scarches",
    "l20_e2_d2_leftout_site4_scarches",
]
trainset.var_names_make_unique()
testset.var_names_make_unique()
scvi.model.MULTIVI.setup_anndata(trainset, batch_key="Site")
scvi.model.MULTIVI.setup_anndata(testset, batch_key="Site")

for i, model_name in enumerate(model_names):
    mvi = scvi.model.MULTIVI.load(model_dir + model_name, adata=trainset)
    rep = mvi.get_latent_representation(testset)
    asw_mvi = silhouette_score(rep, testset.obs["Site"])
    print(asw_mvi)

    temp_df = pd.DataFrame(
        {
            "batch": [batches_left_out[i]],
            "ASW": [asw_mvi],
            "model": ["multiVI + scArches"],
        }
    )
    if i == 0:
        df = temp_df
    else:
        df = pd.concat([df, temp_df])

    model = None
    test_predictions = None
    test_errors = None

df.to_csv("../results/analysis/batch_integration/" + data_name + "_batch_effect_mvi.csv")
print("saved mvi batch effects")