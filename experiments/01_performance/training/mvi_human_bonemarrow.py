# for details about using MultiVI, look at 
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/MultiVI_tutorial.html 
# from where most of this code is

# only the data import and processing is modified and shortened

import os
import scvi
import anndata as ad
import pandas as pd

#random_seed = 0
#random_seed = 8790
random_seed = 37
scvi.settings.seed = random_seed

save_dir = 'results/trained_models/multiVI/human_pbmc/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

adata = ad.read_h5ad('data/human_bonemarrow/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
is_train_df = pd.read_csv('data/human_bonemarrow/train_val_test_split.csv')

train_indices = is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values
adata = adata[train_indices]
adata.var_names_make_unique()
adata.obs['modality'] = 'paired'
adata.X = adata.layers['counts'] # they want unnormalized data in X

model_name = 'l20_e2_d2_rs'+str(random_seed)
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

mvi.train(use_gpu=True)
mvi.save(save_dir+model_name)

elbo = mvi.get_elbo()
print(model_name)
print(elbo)
print(f'Elbo for {model_name} is {elbo}')