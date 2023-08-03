# for details about using MultiVI, look at
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/MultiVI_tutorial.html
# from where most of this code is
# only the data import and processing is modified and shortened
import os
import scvi
import mudata as md
import anndata as ad
import pandas as pd
import numpy as np
import scipy
import torch

###
# define seed in command line
###
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=0)
args = parser.parse_args()
random_seed = args.random_seed
scvi.settings.seed = random_seed
print("training MultiVI on mouse gastrulation data with random seed ", random_seed)

save_dir = "../results/trained_models/multiVI/mouse_gastrulation/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

###
# load data
###
mudata = md.read("../../data/mouse_gastrulation.h5mu", backed=False)
modality_switch = mudata["rna"].X.shape[1]
adata = ad.AnnData(scipy.sparse.hstack((mudata["rna"].X, mudata["atac"].X)))
adata.obs = mudata.obs
mudata = None
adata.var["feature_type"] = "ATAC"
adata.var["feature_type"][:modality_switch] = "GEX"
train_indices = list(np.where(adata.obs["train_val_test"] == "train")[0])
adata = adata[train_indices]
adata.var_names_make_unique()
adata.obs["modality"] = "paired"

###
# set up model
###
model_name = "l20_e2_d2_rs" + str(random_seed)
scvi.model.MULTIVI.setup_anndata(adata, batch_key="stage")
mvi = scvi.model.MULTIVI(
    adata,
    n_genes=modality_switch,
    n_regions=(adata.X.shape[1] - modality_switch),
    n_latent=20,
    n_layers_encoder=2,
    n_layers_decoder=2,
)
mvi.view_anndata_setup()

###
# train
###
import time

start = time.time()
# check if gpu is available
device = True if torch.cuda.is_available() else False
mvi.train(use_gpu=device)
end = time.time()
print("   time: ", end - start)
print("   trained")
mvi.save(save_dir + model_name)
print("   saved")

elbo = mvi.get_elbo()
print(model_name)
print(elbo)
print(f"   Elbo for {model_name} is {elbo}")
