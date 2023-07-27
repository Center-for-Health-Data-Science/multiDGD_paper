# for details about using MultiVI, look at 
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/MultiVI_tutorial.html 
# from where most of this code is

# only the data import and processing is modified and shortened

import os
import scvi
import anndata as ad
import pandas as pd

random_seed = 0
scvi.settings.seed = random_seed
batch_left_out = 3

save_dir = 'results/trained_models/multiVI/human_pbmc/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# get data
adata = ad.read_h5ad('data/human_bonemarrow/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
# select train data without batch to be left out
is_train_df = pd.read_csv('data/human_bonemarrow/train_val_test_split.csv')
batches = adata.obs['Site'].unique()
train_indices_all = list(is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values)
train_indices = [x for x in train_indices_all if adata.obs['Site'].values[x] != batches[batch_left_out]]

adata = adata[train_indices]
adata.var_names_make_unique()
adata.obs['modality'] = 'paired'
adata.X = adata.layers['counts'] # they want unnormalized data in X

model_name = 'l20_e2_d2_leftout_'+batches[batch_left_out]
scvi.model.MULTIVI.setup_anndata(adata, batch_key='Site')
mvi = scvi.model.MULTIVI(
    adata, 
    n_genes=len(adata.var[adata.var['feature_types'] == 'GEX']),
    n_regions=(len(adata.var)-len(adata.var[adata.var['feature_types'] == 'GEX'])),
    #n_hidden=100,
    n_latent=20,
    n_layers_encoder=2,
    n_layers_decoder=2
)
mvi.view_anndata_setup()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ', device)
device_index = device.index
print('device index ', device_index)
mvi.train(use_gpu=device_index)
mvi.save(save_dir+model_name)

elbo = mvi.get_elbo()
print(model_name)
print(elbo)
print(f'Elbo for {model_name} is {elbo}')