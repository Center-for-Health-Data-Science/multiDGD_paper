# for details about using MultiVI, look at
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/MultiVI_tutorial.html
# from where most of this code is

# only the data import and processing is modified and shortened

import os
import scvi
import anndata as ad
import pandas as pd
import numpy as np

# create argument parser for the batch left out
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_left_out", type=int, default=0)
parser.add_argument("--random_seed", type=int, default=0)
args = parser.parse_args()
batch_left_out = args.batch_left_out
random_seed = args.random_seed
scvi.settings.seed = random_seed
print(
    "training MultiVI on human bonemarrow data with random seed ",
    random_seed,
    " and batch left out: ",
    batch_left_out,
)

save_dir = "../results/trained_models/multiVI/human_bonemarrow/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

###
# load data
###
data_name = "human_bonemarrow"
adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
adata.X = adata.layers["counts"]
batches = adata.obs["Site"].unique()
train_indices_all = list(np.where(adata.obs["train_val_test"] == "train")[0])
train_indices = [
    x
    for x in train_indices_all
    if adata.obs["Site"].values[x] != batches[batch_left_out]
]
adata = adata[train_indices]
adata.var_names_make_unique()
adata.obs["modality"] = "paired"

model_name = "l20_e2_d2_leftout_" + batches[batch_left_out]
scvi.model.MULTIVI.setup_anndata(adata, batch_key="Site")
mvi = scvi.model.MULTIVI(
    adata,
    n_genes=len(adata.var[adata.var["feature_types"] == "GEX"]),
    n_regions=(len(adata.var) - len(adata.var[adata.var["feature_types"] == "GEX"])),
    n_latent=20,
    n_layers_encoder=2,
    n_layers_decoder=2,
)
mvi.view_anndata_setup()

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_index = device.index
mvi.train(use_gpu=device_index)
mvi.save(save_dir + model_name)

elbo = mvi.get_elbo()
print(f"   Elbo for {model_name} is {elbo}")
