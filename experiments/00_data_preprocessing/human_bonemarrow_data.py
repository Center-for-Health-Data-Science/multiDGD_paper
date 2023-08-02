############################
# preparing the human bone marrow data
# the result is the anndata object on figshare
############################
print("preparing the human bonemarrow data")

# imports
import anndata as ad
import numpy as np
import pandas as pd

###
# load data
###
print("   loading data")
adata = ad.read_h5ad(
    "../../data/raw/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
)

###
# make train-val-test split
###
dataset_ids = np.arange(adata.shape[0])
np.random.shuffle(dataset_ids)
n_testsamples = int(0.1 * adata.shape[0])
train_val_test_split = pd.DataFrame(
    {"num_idx": np.arange(adata.shape[0]), "is_train": "train"}
)
train_val_test_split["is_train"].values[dataset_ids[:n_testsamples]] = "validation"
train_val_test_split["is_train"].values[
    dataset_ids[n_testsamples : (2 * n_testsamples)]
] = "test"

###
# prep obs for data
###
adata.obs["cell_type"]
adata.obs["observable"] = adata.obs["cell_type"].values
adata.obs["covariate_Site"] = adata.obs["Site"].values
adata.X = adata.layers["counts"]

###
# save data
###
adata.write_h5ad("../../data/human_bonemarrow.h5ad")
print("   saved data")
