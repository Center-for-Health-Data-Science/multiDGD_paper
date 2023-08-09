##############
# imports
##############

import anndata as ad
import pandas as pd
import numpy as np
import scvi

from omicsdgd import DGD

##############
# load data
##############
fraction_unpaired = 0
random_seed = 0
save_dir = "results/trained_models/"
data_name = "human_bonemarrow"
adata = ad.read_h5ad("data/" + data_name + ".h5ad")
train_indices = np.where(adata.obs["train_val_test"] == "train")[0]
trainset = adata[train_indices, :].copy()
test_indices = np.where(adata.obs["train_val_test"] == "test")[0]
adata_test = adata[test_indices, :].copy()

##############
# load model
##############
model = DGD.load(
    data=trainset,
    save_dir=save_dir + data_name + "/",
    model_name=data_name + "_l20_h2-3_rs" + str(random_seed),
)
# change the model name so that the original test representations will not be overwritten
model._model_name = "human_bonemarrow_l20_h2-3_rs0_unpaired0percent"
print("   loaded")

adata_rna = adata_test[:, adata_test.var["feature_types"] == "GEX"].copy()
adata_atac = adata_test[:, adata_test.var["feature_types"] == "ATAC"].copy()
adata = None
adata_unpaired_test = scvi.data.organize_multiome_anndatas(
    adata_test, adata_rna, adata_atac
)
print("   organized data")
adata_rna, adata_atac, adata_test = None, None, None

model.predict_new(adata_unpaired_test, n_epochs=50)
print("   new samples learned")

###
# extract latent representations
###
rep = model.test_rep.z.detach().numpy()
cell_labels = adata_unpaired_test.obs["cell_type"].values
batch_labels = adata_unpaired_test.obs["Site"].values
modality_labels = adata_unpaired_test.obs["modality"].values
cell_indices = [x.split("_")[0] for x in adata_unpaired_test.obs_names.values]

rep_df = pd.DataFrame(rep, columns=np.arange(rep.shape[1]))
rep_df["cell_type"] = cell_labels
rep_df["batch"] = batch_labels
rep_df["modality"] = modality_labels
rep_df["cell_idx"] = cell_indices
rep_df.to_csv(
    "results/analysis/modality_integration/human_bonemarrow_l20_h2-3_rs0_unpaired0percent_testlatent.csv"
)

# now make a umap and export it
import umap.umap_ as umap

# the umap should be fitted to the training representations and map the test representations
# labels will be ['train', 'test (paired)', 'test (atac)', 'test (rna)']

# first, get the training representations
train_rep = model.representation.z.detach().numpy()
# fit a umap
n_neighbors = 50
min_dist = 0.75
column_names = ['UMAP1', 'UMAP2']
reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist)
projected = reducer.fit_transform(train_rep)
# map the test representations
projected_test = reducer.transform(rep)
# make a dataframe
umap_df = pd.DataFrame(projected, columns=column_names)
umap_df["data_set"] = "train"
umap_df_test = pd.DataFrame(projected_test, columns=column_names)
umap_df_test["data_set"] = modality_labels
# replace every 'expression' with 'rna' and 'accessibility' with 'atac'
umap_df_test["data_set"] = umap_df_test["data_set"].replace("paired", "test (paired)")
umap_df_test["data_set"] = umap_df_test["data_set"].replace("expression", "test (rna)")
umap_df_test["data_set"] = umap_df_test["data_set"].replace(
    "accessibility", "test (atac)"
)
# save umap
umap_df.to_csv(
    "results/analysis/modality_integration/human_bonemarrow_l20_h2-3_rs0_unpaired0percent_latent_integration_umap_all.csv"
)
