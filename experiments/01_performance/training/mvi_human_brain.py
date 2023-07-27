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

#random_seed = 0
random_seed = 8790
#random_seed = 37
scvi.settings.seed = random_seed

save_dir = 'results/trained_models/multiVI/human_brain/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mudata = md.read('data/human_brain/mudata.h5mu', backed=False)
modality_switch = mudata['rna'].X.shape[1]
adata = ad.AnnData(scipy.sparse.hstack((mudata['rna'].X,mudata['atac'].X)))
adata.obs['celltype'] = mudata.obs['celltype']
adata.var['feature_type'] = 'ATAC'
adata.var['feature_type'][:modality_switch] = 'GEX'
mudata = None
adata.X = adata.X.tocsr()
#adata.layers['counts'] = adata.X

#mudata.obs['stage'] = mudata['atac'].obs['stage']
is_train_df = pd.read_csv('data/human_brain/train_val_test_split.csv')

train_indices = is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values
adata = adata[train_indices]
#adata.var_names_make_unique()
adata.obs['modality'] = 'paired'
#adata.X = adata.layers['counts'] # they want unnormalized data in X

model_name = 'l20_e1_d1_rs'+str(random_seed)
scvi.model.MULTIVI.setup_anndata(adata)
mvi = scvi.model.MULTIVI(
    adata, 
    n_genes=modality_switch,
    n_regions=(adata.X.shape[1]-modality_switch),
    n_latent=20,
    n_layers_encoder=1,
    n_layers_decoder=1
)
mvi.view_anndata_setup()

mvi.train(use_gpu=True)
print('trained')
mvi.save(save_dir+model_name)
print('saved')

elbo = mvi.get_elbo()
print(model_name)
print(elbo)
print(f'Elbo for {model_name} is {elbo}')