import scvi
import pandas as pd
from omicsdgd import DGD
import numpy as np
import anndata as ad
import mudata as md
import torch
import scanpy as sc
from sklearn import preprocessing
from sklearn.metrics import adjusted_rand_score, silhouette_score
from omicsdgd.functions._analysis import (
    testset_reconstruction_evaluation,
    gmm_clustering,
)


def cluster_dgd(mod, norm=True):
    classes = list(mod.train_set.meta.unique())
    true_labels = np.asarray([classes.index(i) for i in mod.train_set.meta])
    cluster_labels = gmm_clustering(mod.representation, mod.gmm, mod.train_set.meta)
    return cluster_labels


# define seed in command line
import argparse

parser = argparse.ArgumentParser()
help_string = (
    "index of data to be used:\n   0: bonemarrow\n   1: gastrulation\n   2: brain"
)
parser.add_argument("--data_index", type=int, default=0, help=help_string)
args = parser.parse_args()
data_index = args.data_index

save_dir = "results/trained_models/"
data_names = ["human_bonemarrow", "mouse_gastrulation", "human_brain"]
model_names_all = [
    [
        ["l20_e2_d2", "l20_e2_d2_rs37", "l20_e2_d2_rs8790"],
        [
            "human_bonemarrow_l20_h2-3",
            "human_bonemarrow_l20_h2-3_rs37",
            "human_bonemarrow_l20_h2-3_rs8790",
        ],
    ],
    [
        ["l20_e2_d2", "l20_e2_d2_rs37", "l20_e2_d2_rs8790"],
        [
            "mouse_gast_l20_h2-2_rs0",
            "mouse_gast_l20_h2-2_rs37",
            "mouse_gast_l20_h2-2_rs8790",
        ],
    ],
    [
        ["l20_e1_d1", "l20_e1_d1_rs37", "l20_e1_d1_rs8790"],
        [
            "human_brain_l20_h2-2_a2_long",
            "human_brain_l20_h2-2_a2_rs37",
            "human_brain_l20_h2-2_a2_rs8790",
        ],
    ],
]
data_name = data_names[data_index]
model_names = model_names_all[data_index]
random_seeds = [0, 37, 8790]
plotting_keys = ["cell_type", "celltype", "celltype"]
leiden_resolutions = [1, 2, 1]
correction = ["Site", "stage", None]

"""
Go through datasets and chosen models and compute reconstruction performances
"""

if data_name == "human_bonemarrow":
    data = ad.read_h5ad("data/human_bonemarrow.h5ad")
    modality_switch = 13431
else:
    import scipy

    if data_name == "mouse_gastrulation":
        mdata = md.read("data/mouse_gastrulation.h5ad", backed=False)
    elif data_name == "human_brain":
        mdata = md.read("data/human_brain.h5ad", backed=False)
    data = ad.AnnData(scipy.sparse.hstack((mdata["rna"].X, mdata["atac"].X)))
    data.obs = mdata.obs
    modality_switch = mdata["rna"].shape[1]
    mdata = None
train_indices = list(np.where(data.obs["train_val_test"] == "train")[0])
test_indices = list(np.where(data.obs["train_val_test"] == "test")[0])
trainset = data[train_indices, :]
testset = data[test_indices, :]
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
print("   loaded data")
# get true labels for clustering ARI later
le = preprocessing.LabelEncoder()
le.fit(trainset.obs[plotting_keys[data_index]].values)
true_labels = le.transform(trainset.obs[plotting_keys[data_index]].values)

# multiDGD first
for count, model_name in enumerate(model_names[1]):
    print(model_name)

    # compute for DGD
    model = DGD.load(
        data=trainset, save_dir=save_dir + data_name + "/", model_name=model_name
    )
    print("loaded model")
    if data_name == "human_brain":
        metrics_temp = testset_reconstruction_evaluation(
            testset, model, modality_switch, library, thresholds=[0.2], batch_size=32
        )
    else:
        metrics_temp = testset_reconstruction_evaluation(
            testset, model, modality_switch, library, thresholds=[0.2]
        )
    metrics_temp["model"] = "multiDGD"
    metrics_temp["random_seed"] = random_seeds[count]
    # now ARI and ASW
    # ARI
    cluster_labels = cluster_dgd(model)
    radj = adjusted_rand_score(true_labels, np.asarray(cluster_labels))
    # ASW
    asw = None
    if correction[data_index] is not None:
        data.obsm["latent"] = model.get_latent_representation()
        asw = silhouette_score(data.obsm["latent"], data.obs[correction])
    model = None
    metrics_temp["ARI"] = radj
    metrics_temp["ASW"] = asw
    if count == 0:
        metrics_dgd = metrics_temp
    else:
        metrics_dgd = metrics_dgd.append(metrics_temp)

# compute for multiVI
if data_name == "mouse_gastrulation":
    trainset.var_names_make_unique()
    trainset.obs["modality"] = "paired"
    # trainset.obs['_indices'] = np.arange(trainset.n_obs)
    scvi.model.MULTIVI.setup_anndata(trainset, batch_key="stage")
    testset.var_names_make_unique()
    testset.obs["modality"] = "paired"
    # testset.obs['_indices'] = np.arange(testset.n_obs)
    scvi.model.MULTIVI.setup_anndata(testset, batch_key="stage")
elif data_name == "human_bonemarrow":
    trainset.var_names_make_unique()
    trainset.obs["modality"] = "paired"
    scvi.model.MULTIVI.setup_anndata(trainset, batch_key="Site")
    testset.var_names_make_unique()
    testset.obs["modality"] = "paired"
    scvi.model.MULTIVI.setup_anndata(testset, batch_key="Site")
else:
    trainset.var_names_make_unique()
    trainset.obs["modality"] = "paired"
    scvi.model.MULTIVI.setup_anndata(trainset)
    testset.var_names_make_unique()
    testset.obs["modality"] = "paired"
    scvi.model.MULTIVI.setup_anndata(testset)

for count, model_name in enumerate(model_names[0]):
    print(model_name)
    model = scvi.model.MULTIVI.load(
        save_dir + "multiVI/" + data_name + "/" + model_name, adata=trainset
    )
    if data_name == "human_brain":
        metrics_temp = testset_reconstruction_evaluation(
            testset, model, modality_switch, library, batch_size=32
        )
    else:
        metrics_temp = testset_reconstruction_evaluation(
            testset, model, modality_switch, library
        )
    metrics_temp["model"] = "multiVI"
    metrics_temp["random_seed"] = random_seeds[count]
    # now ARI and ASW
    # ARI
    sc.pp.neighbors(trainset, use_rep="latent", n_neighbors=15)
    sc.tl.leiden(
        trainset, key_added="clusters", resolution=leiden_resolutions[data_index]
    )
    cluster_labels = trainset.obs["clusters"].values.astype(int)
    radj = adjusted_rand_score(true_labels, np.asarray(cluster_labels))
    # ASW
    asw = None
    if correction[data_index] is not None:
        data.obsm["latent"] = model.get_latent_representation()
        asw = silhouette_score(data.obsm["latent"], data.obs[correction])
    model = None
    metrics_temp["ARI"] = radj
    metrics_temp["ASW"] = asw
    if count == 0:
        metrics_mvi = metrics_temp
    else:
        metrics_mvi = metrics_mvi.append(metrics_temp)

metrics_df = pd.concat([metrics_mvi, metrics_dgd])
metrics_df.to_csv(
    "results/analysis/performance_evaluation/reconstruction/" + data_name + ".csv"
)

print("   done")
