############################
# preparing the mouse gastrulation data
# the result is the mudata object on figshare
############################
print("preparing the mouse gastrulation data")

# imports
import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

###
# load data
###
print("   loading data")
gex = ad.read_h5ad("../../data/raw/mouse_gastrulation_anndata.h5ad")
atac = ad.read_h5ad("../../data/raw/mouse_gastrulation_PeakMatrix_anndata.h5ad")

###
# pair the data
###
# find intersect of sampled that passed both QCs
ids_shared = list(
    set(gex.obs["sample"].index.values).intersection(
        set(atac.obs["sample"].index.values)
    )
)
# get data indices for both sets
ids_gex = np.where(gex.obs["sample"].index.isin(ids_shared))[0]
ids_atac = np.where(atac.obs["sample"].index.isin(ids_shared))[0]
# subset datasets
gex = gex[ids_gex]
atac = atac[ids_atac]

###
# do feature selection (nonzero in 0.5% of samples)
###
print("   feature selection")
threshold = 0.05
percent_threshold = int(threshold * len(ids_shared))
gene_nonzero_id, gene_nonzero_count = np.unique(
    gex.X.copy().tocsr().nonzero()[1], return_counts=True
)
selected_features = gene_nonzero_id[
    np.where(gene_nonzero_count >= percent_threshold)[0]
]
modality_switch = len(selected_features)
print("selected " + str(len(selected_features)) + " gex features")
threshold = 0.05
percent_threshold = int(threshold * len(ids_shared))
atac_nonzero_id, atac_nonzero_count = np.unique(
    atac.X.copy().tocsr().nonzero()[1], return_counts=True
)
selected_features_atac = atac_nonzero_id[
    np.where(atac_nonzero_count >= percent_threshold)[0]
]
print("selected " + str(len(selected_features_atac)) + " atac features")

###
# make mudata object
###
print("   creating mudata object")
mudata = md.MuData(
    {"rna": gex[:, selected_features], "atac": atac[:, selected_features_atac]}
)
# mudata.write("data/mouse_gastrulation/mudata.h5mu")

###
# make train-val-test split
###
dataset_ids = np.arange(len(ids_shared))
np.random.shuffle(dataset_ids)
n_testsamples = int(0.1 * len(ids_shared))
train_val_test_split = pd.DataFrame(
    {"num_idx": np.arange(len(ids_shared)), "is_train": "train"}
)
train_val_test_split["is_train"].values[dataset_ids[:n_testsamples]] = "validation"
train_val_test_split["is_train"].values[
    dataset_ids[n_testsamples : (2 * n_testsamples)]
] = "test"

###
# prep obs for data
###
mudata.obs["train_val_test"] = train_val_test_split["is_train"].values
mudata.obs["celltype"] = mudata["rna"].obs["celltype"].values
mudata.obs["stage"] = mudata["rna"].obs["stage"].values
mudata.obs["observable"] = mudata.obs["train_val_test"].values
mudata.obs["covariate_stage"] = mudata.obs["stage"].values

###
# save data
###
mudata.write("../../data/mouse_gastrulation.h5mu")
print("   saved data")
