# for details about using MultiVI, look at 
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/MultiVI_tutorial.html 
# from where most of this code is

# only the data import and processing is modified and shortened

import os
import scvi
import anndata as ad
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--fraction', type=float)
args = parser.parse_args()
random_seed = args.random_seed
fraction = args.fraction
#fraction_options = [0.01, 0.1, 0.25, 0.5, 0.75]
#fraction = 0.01

#random_seed = 0
#random_seed = 37
#random_seed = 8790
scvi.settings.seed = random_seed

data_name = 'human_bonemarrow'
save_dir = 'results/trained_models/multiVI/human_bonemarrow/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# get subset of train indices
df_subset_ids = pd.read_csv('data/'+data_name+'/data_subsets.csv')
train_indices = list(df_subset_ids[(df_subset_ids['fraction'] == fraction) & (df_subset_ids['include'] == 1)]['sample_idx'].values)
n_samples = len(train_indices)

# get data
adata = ad.read_h5ad('data/'+data_name+'/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
adata = adata[train_indices]
adata.var_names_make_unique()
adata.X = adata.layers['counts'] # they want unnormalized data in X

model_name = 'l20_e2_d2_rs'+str(random_seed)+'_subset'+str(n_samples)
scvi.model.MULTIVI.setup_anndata(adata, batch_key='Site')
mvi = scvi.model.MULTIVI(
    adata, 
    n_genes=len(adata.var[adata.var['feature_types'] == 'GEX']),
    n_regions=(len(adata.var)-len(adata.var[adata.var['feature_types'] == 'GEX'])),
    n_latent=20,
    n_layers_encoder=2,
    n_layers_decoder=2
)
mvi.view_anndata_setup()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_index = device.index
mvi.train(use_gpu=device_index)
mvi.save(save_dir+model_name)

elbo = mvi.get_elbo()
print(model_name)
print(elbo)
print(f'Elbo for {model_name} is {elbo}')