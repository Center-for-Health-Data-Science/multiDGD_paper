##############
# imports
##############

import anndata as ad
import pandas as pd
import numpy as np
import scvi

from omicsdgd import DGD

##############
# define fraction of unpaired cells
##############
#fraction_unpaired_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
fraction_unpaired = 0

##############
# load data
##############

save_dir = 'results/trained_models/'

data_name = 'human_bonemarrow'
is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
df_unpaired = pd.read_csv('data/'+data_name+'/unpairing.csv')
adata = ad.read_h5ad('data/'+data_name+'/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
# make test set now to be able to free memory earlier
adata_test = adata[is_train_df[is_train_df['is_train'] == 'iid_holdout']['num_idx'].values,:].copy()
print(adata_test)
print('data loaded')

# train-validation-test split for reproducibility

train_indices = is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values


if fraction_unpaired > 0.0:
    mod_1_indices = df_unpaired[(df_unpaired['fraction_unpaired'] == fraction_unpaired) & (df_unpaired['modality'] == 'rna')]['sample_idx'].values
    mod_2_indices = df_unpaired[(df_unpaired['fraction_unpaired'] == fraction_unpaired) & (df_unpaired['modality'] == 'atac')]['sample_idx'].values
    if fraction_unpaired == 1.0:
        adata_unpaired = adata[mod_1_indices,:].copy()
        adata_unpaired.obs['modality'] = 'GEX'
        adata_temp = adata[mod_2_indices,:].copy()
        adata_temp.obs['modality'] = 'ATAC'
        adata_unpaired = adata_unpaired.concatenate(adata_temp)
        print("organized data")
        adata, adata_temp = None, None
    else:
        remaining_indices = df_unpaired[(df_unpaired['fraction_unpaired'] == fraction_unpaired) & (df_unpaired['modality'] == 'paired')]['sample_idx'].values
        print('made indices')
        adata_rna = adata[mod_1_indices, adata.var['feature_types'] == 'GEX'].copy()
        print('copied rna')
        adata_atac = adata[mod_2_indices, adata.var['feature_types'] == 'ATAC'].copy()
        print('copied atac')
        adata_multi = adata[remaining_indices, :].copy()
        print('copied rest')
        adata = None
        adata_unpaired = scvi.data.organize_multiome_anndatas(adata_multi, adata_rna, adata_atac)
        print('organized data')
        adata_rna, adata_atac, adata_multi = None, None, None

##############
# initialize model
##############
hyperparameters = {
    'latent_dimension': 20,
    'n_hidden': 2,
    'n_hidden_modality': 3,
    'log_wandb': ['viktoriaschuster', 'omicsDGD']
}
#random_seed = 8790
#random_seed = 37
random_seed = 0
dgd_name = data_name+'_l20_h2-3_rs'+str(random_seed)+'_unpaired'+str(int(fraction_unpaired*100))+'percent'
#dgd_name = 'human_bonemarrow_l20_h2-3_test2'

if fraction_unpaired > 0.0:
    model = DGD.load(data=adata_unpaired,
                save_dir=save_dir+data_name+'/',
                model_name=dgd_name)
else:
    model = DGD.load(data=adata[train_indices,:],
                save_dir=save_dir+data_name+'/',
                model_name=dgd_name)
print('loaded model')

adata_rna = adata_test[:, adata_test.var['feature_types'] == 'GEX'].copy()
print('copied rna')
adata_atac = adata_test[:, adata_test.var['feature_types'] == 'ATAC'].copy()
print('copied atac')
adata = None
adata_unpaired_test = scvi.data.organize_multiome_anndatas(adata_test, adata_rna, adata_atac)
print(adata_unpaired_test)
print('organized data')
adata_rna, adata_atac, adata_test = None, None, None

model.predict_new(adata_unpaired_test, n_epochs=50)
print('new samples learned')