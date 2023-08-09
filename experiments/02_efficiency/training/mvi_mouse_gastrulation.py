# for details about using MultiVI, look at
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/MultiVI_tutorial.html
# from where most of this code is

# only the data import and processing is modified and shortened

import os
import scvi
import mudata as md
import anndata as ad
import pandas as pd
import scipy

random_seed = 0
scvi.settings.seed = random_seed

save_dir = "results/trained_models/multiVI/mouse_gastrulation/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# test different feature selection thresholds
import anndata as ad
import numpy as np

gex = ad.read_h5ad("data/raw/anndata.h5ad")
atac = ad.read_h5ad("data/raw/PeakMatrix_anndata.h5ad")
ids_shared = list(
    set(gex.obs["sample"].index.values).intersection(
        set(atac.obs["sample"].index.values)
    )
)
ids_gex = np.where(gex.obs["sample"].index.isin(ids_shared))[0]
ids_atac = np.where(atac.obs["sample"].index.isin(ids_shared))[0]
gex = gex[ids_gex]
atac = atac[ids_atac]
threshold = 0.00
if threshold > 0.000:
    percent_threshold = int(threshold * len(ids_shared))
    gene_nonzero_id, gene_nonzero_count = np.unique(
        gex.X.copy().tocsr().nonzero()[1], return_counts=True
    )
    selected_features = gene_nonzero_id[
        np.where(gene_nonzero_count >= percent_threshold)[0]
    ]
    modality_switch = len(selected_features)
    print("selected " + str(len(selected_features)) + " gex features")
    percent_threshold = int(threshold * len(ids_shared))
    atac_nonzero_id, atac_nonzero_count = np.unique(
        atac.X.copy().tocsr().nonzero()[1], return_counts=True
    )
    selected_features_atac = atac_nonzero_id[
        np.where(atac_nonzero_count >= percent_threshold)[0]
    ]
    print("selected " + str(len(selected_features_atac)) + " atac features")
    mudata = md.MuData(
        {"rna": gex[:, selected_features], "atac": atac[:, selected_features_atac]}
    )
else:
    mudata = md.MuData({"rna": gex, "atac": atac})

# mudata = md.read('data/mouse_gastrulation/mudata.h5mu', backed=False)
modality_switch = mudata["rna"].X.shape[1]
adata = ad.AnnData(scipy.sparse.hstack((mudata["rna"].X, mudata["atac"].X)))
adata.obs["celltype"] = mudata["rna"].obs["celltype"]
adata.obs["stage"] = mudata["atac"].obs["stage"].values
adata.var["feature_type"] = "ATAC"
adata.var["feature_type"][:modality_switch] = "GEX"
mudata = None
train_indices = np.where(mudata.obs["train_val_test"] == "train")[0]
adata = adata[train_indices]
adata.var_names_make_unique()
adata.obs["modality"] = "paired"

model_name = (
    "l20_e2_d2_rs" + str(random_seed) + "_featselect" + str(threshold).split(".")[1]
)
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

import time

start = time.time()
mvi.train(use_gpu=True)
end = time.time()
print("   time: ", end - start)
print("   trained")
mvi.save(save_dir + model_name)
print("   saved")

elbo = mvi.get_elbo()
print(model_name)
print(elbo)
print(f"   Elbo for {model_name} is {elbo}")
