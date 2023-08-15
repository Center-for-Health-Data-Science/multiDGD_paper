import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import math
import anndata as ad
import mudata as md

save_dir = '../results/trained_models/'
data_name = 'mouse_gastrulation'
random_seed = 0

# first load data

# load data (full set)
gex = ad.read_h5ad("../../data/raw/mouse_gastrulation_anndata.h5ad")
#modality_switch = gex.X.shape[1]
atac = ad.read_h5ad("../../data/raw/mouse_gastrulation_PeakMatrix_anndata.h5ad")
ids_shared = list(
    set(gex.obs["sample"].index.values).intersection(
        set(atac.obs["sample"].index.values)
    )
)
ids_gex = np.where(gex.obs["sample"].index.isin(ids_shared))[0]
ids_atac = np.where(atac.obs["sample"].index.isin(ids_shared))[0]
gex = gex[ids_gex]
atac = atac[ids_atac]
modality_switch_full = gex.X.shape[1]

# load data (feature selected)
mudata = md.read("../../data/mouse_gastrulation.h5mu", backed=False)

# get test indices
test_indices = np.where(mudata.obs["train_val_test"] == "test")[0]
gex_test = gex[test_indices]
atac_test = atac[test_indices]
mudata_test = mudata[test_indices]

# get the indices of the features that overlap between the sets
rna_indices = [i for i, x in enumerate(gex_test.var.index) if x in mudata['rna'].var.index]
atac_indices = [i-modality_switch_full for i, x in enumerate(atac_test.var.index) if x in mudata['atac'].var.index]
# save indices as csv file
indices_df = pd.concat(
    (pd.DataFrame({'idx': rna_indices,
                           'modality': 'rna'}),
    pd.DataFrame({'idx': atac_indices,
                           'modality': 'atac'})), axis=0
)
# make idx column integer
indices_df['idx'] = indices_df['idx'].astype(int)
indices_df.to_csv('../../data/mouse_gastrulation_five_percent_indices.csv')

# save test data for full and feature-selected set
np.save("../results/analysis/performance_evaluation/reconstruction/mouse_gast_test_counts_gex.npy", np.asarray(gex_test.X[:,rna_indices]))
np.save("../results/analysis/performance_evaluation/reconstruction/mouse_gast_test_counts_atac.npy", np.asarray(atac_test.X[:,atac_indices]))