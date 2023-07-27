##############
# imports
##############

import anndata as ad
import pandas as pd
import numpy as np
import scvi

from omicsdgd import DGD

##############
# define fraction of unpaired cells
##############
# fraction_unpaired_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
fraction_unpaired = 0.5

##############
# load data
##############

data_name = "human_bonemarrow"
is_train_df = pd.read_csv("data/" + data_name + "/train_val_test_split.csv")
df_unpaired = pd.read_csv("data/" + data_name + "/unpairing.csv")
adata = ad.read_h5ad("data/" + data_name + "/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")
# make test set now to be able to free memory earlier
adata_test = adata[is_train_df[is_train_df["is_train"] == "test"]["num_idx"].values, :].copy()
adata_test.obs['modality'] = 'paired'
print("data loaded")

# train-validation-test split for reproducibility

train_indices = is_train_df[is_train_df["is_train"] == "train"]["num_idx"].values

mod_1_indices = df_unpaired[
    (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "rna")
]["sample_idx"].values
mod_2_indices = df_unpaired[
    (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "atac")
]["sample_idx"].values
remaining_indices = df_unpaired[
    (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "paired")
]["sample_idx"].values
print("made indices")

if fraction_unpaired < 1.0:
    adata_rna = adata[mod_1_indices, adata.var["feature_types"] == "GEX"].copy()
    print("copied rna")
    adata_atac = adata[mod_2_indices, adata.var["feature_types"] == "ATAC"].copy()
    print("copied atac")
    adata_multi = adata[remaining_indices, :].copy()
    print("copied rest")
    adata = None
    print("freed some memory")
    adata_unpaired = scvi.data.organize_multiome_anndatas(adata_multi, adata_rna, adata_atac)
    print("organized data")
else:
    adata_unpaired = adata[mod_1_indices,:].copy()
    adata_unpaired.obs['modality'] = 'GEX'
    adata_temp = adata[mod_2_indices,:].copy()
    adata_temp.obs['modality'] = 'ATAC'
    adata_unpaired = adata_unpaired.concatenate(adata_temp)
    print("organized data")
    adata, adata_temp = None, None

adata_rna, adata_atac, adata_multi = None, None, None
print("freed memory")
adata = adata_unpaired.concatenate(adata_test)
print("finished data")

# train-validation-test split for reproducibility
# best provided as list [[train_indices], [validation_indices]]
train_val_split = [
    list(np.arange(len(train_indices))),
    list(np.arange(len(train_indices), len(train_indices) + len(adata_test))),
]
# print('prepared data')

##############
# initialize model
##############
hyperparameters = {
    "latent_dimension": 20,
    "n_hidden": 2,
    "n_hidden_modality": 3,
    "log_wandb": ["viktoriaschuster", "omicsDGD"],
}
# random_seed = 8790
# random_seed = 37
random_seed = 0

model = DGD(
    data=adata,
    parameter_dictionary=hyperparameters,
    train_validation_split=train_val_split,
    modalities="feature_types",
    meta_label="cell_type",
    correction="Site",
    save_dir="./results/trained_models/" + data_name + "/",
    model_name=data_name
    + "_l20_h2-3_rs"
    + str(random_seed)
    + "_unpaired"
    + str(int(fraction_unpaired * 100))
    + "percent"
    + "_10pPairedTriangleLoss2",
    random_seed=random_seed,
)
print("initialized model")

###
# train and save
###

model.train(n_epochs=1000, train_minimum=100, developer_mode=True, stop_after=20)
# train technically needs no input, but specifying how long it should be trained and what the desired performance metric is helps get the best results

model.save()
print("model saved")
exit()
