# for details about using MultiVI, look at
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/MultiVI_tutorial.html
# from where most of this code is
# only the data import and processing is modified and shortened

import os
import scvi
import anndata as ad
import pandas as pd
import numpy as np
import torch

# define seed in command line
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=0)
args = parser.parse_args()
random_seed = args.random_seed
scvi.settings.seed = random_seed
print("training MultiVI on human bonemarrow data with random seed ", random_seed)

save_dir = "../results/trained_models/multiVI/human_bonemarrow/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_name = "human_bonemarrow"
adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")

train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
adata = adata[train_indices]
adata.var_names_make_unique()
adata.obs["modality"] = "paired"

model_name = "l20_e2_d2_rs" + str(random_seed)
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
# check if gpu is available
device = True if torch.cuda.is_available() else False
mvi.train(use_gpu=device)
mvi.save(save_dir + model_name)

elbo = mvi.get_elbo()
print(model_name)
print(elbo)
print(f"   Elbo for {model_name} is {elbo}")
