"""
scArches has no support yet for MultiVI,
si I do this for scVI with only expression data

load a trained multiVI (currently scVI) model and integrate a new 'batch' of samples by using scArches
"""
import scvi
import anndata as ad
import pandas as pd
import numpy as np

# import scarches as sca

# create argument parser for the batch left out
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_left_out", type=int, default=0)
parser.add_argument("--random_seed", type=int, default=0)
args = parser.parse_args()
batch_left_out = args.batch_left_out
random_seed = args.random_seed
scvi.settings.seed = random_seed

###
# load data
###
data_name = "human_bonemarrow"
adata = ad.read_h5ad("data/" + data_name + ".h5ad")
batches = adata.obs["Site"].unique()
train_indices_all = list(np.where(adata.obs["train_val_test"] == "train")[0])
train_indices = [
    x
    for x in train_indices_all
    if adata.obs["Site"].values[x] != batches[batch_left_out]
]
test_indices = list(np.where(adata.obs["train_val_test"] == "test")[0])
adata = adata[train_indices]
adata.var_names_make_unique()
adata.obs["modality"] = "paired"

model_name = "l20_e2_d2_leftout_" + batches[batch_left_out]

################
# load trained model
################
adata_train = adata.copy()[train_indices]
scvi.model.MULTIVI.setup_anndata(adata_train, batch_key="Site")

################
# load trained model
################
print("load model")
model_dir = "results/trained_models/multiVI/human_pbmc/"
model_name = "l20_e2_d2_leftout_" + batches[batch_left_out]

################
# apply scArches
################
print("apply scArches")
# see tutorial for more https://scarches.readthedocs.io/en/latest/scvi_surgery_pipeline.html#
adata_test = adata.copy()[test_indices]
scvi.model.MULTIVI.prepare_query_anndata(adata_test, model_dir + model_name)
print("prepared query anndata")
model = scvi.model.MULTIVI.load_query_data(adata_test, model_dir + model_name)
print("initialized new model")
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_index = device.index
model.train(max_epochs=200, plan_kwargs=dict(weight_decay=0.0), use_gpu=device_index)

# save new model
model.save(model_dir + model_name + "_scarches")
