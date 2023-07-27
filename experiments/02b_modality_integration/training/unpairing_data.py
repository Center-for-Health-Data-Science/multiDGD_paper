'''
This script creates a dataframe with a column of selected modalities for each fraction of unpaired cells
the modality will be either `paired`, `rna` or `atac`
'''

##############
# imports
##############

import pandas as pd
import numpy as np

##############
# define fraction of unpaired cells
##############
# I think there might be a problem with unpairing data and keeping both modalities
# because this increases the number of cells in the dataset significantly
# alternatively, I can keep only one modality of the unpaired cells (alternating)
# and measure the performance on all test samples unpaired with both modalities and 
# computing the distances between representations of the same cell
fraction_unpaired_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# get train test split to know how many samples are in the train set
data_name = 'human_bonemarrow'
is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
train_indices = is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values
n_samples = len(train_indices)
# create out dataframe for unpairing == 0
df_out = pd.DataFrame({
    'sample_idx': train_indices,
    'fraction_unpaired': [0]*n_samples,
    'modality': ['paired']*n_samples
})
for fraction_unpaired in fraction_unpaired_options:
    # select random samples to be unpaired
    unpairing = np.random.choice(np.arange(n_samples), size=int(len(train_indices)*fraction_unpaired), replace=False)
    mod_1_indices = unpairing[::2]
    mod_2_indices = unpairing[1::2]
    # make the string list
    modality = ['paired']*n_samples
    modality = ['rna' if i in mod_1_indices else x for i,x in enumerate(modality)]
    modality = ['atac' if i in mod_2_indices else x for i,x in enumerate(modality)]
    # make temp df
    df_temp = pd.DataFrame({
        'sample_idx': train_indices,
        'fraction_unpaired': [fraction_unpaired]*n_samples,
        'modality': modality
    })
    df_out = pd.concat([df_out, df_temp], axis=0)
df_out.to_csv('data/'+data_name+'/unpairing.csv', index=False)

"""
##############
# load data
##############

data_name = 'human_bonemarrow'
adata = ad.read_h5ad('data/'+data_name+'/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')

for fraction_unpaired in fraction_unpaired_options:
    print('fraction unpaired: '+str(fraction_unpaired))
    # train-validation-test split for reproducibility
    is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
    train_indices = is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values

    # generate a random subset of the train indices based on the defined fraction
    unimodal_indices = np.random.choice(train_indices, size=int(len(train_indices)*fraction_unpaired), replace=False)
    mod_1_indices = unimodal_indices[::2]
    mod_2_indices = unimodal_indices[1::2]
    remaining_indices = np.setdiff1d(train_indices, unimodal_indices)

    adata_rna = adata[mod_1_indices, adata.var['feature_types'] == 'GEX'].copy()
    adata_atac = adata[mod_2_indices, adata.var['feature_types'] == 'ATAC'].copy()
    adata_multi = adata[remaining_indices, :].copy()

    adata_unpaired = scvi.data.organize_multiome_anndatas(adata_multi, adata_rna, adata_atac)
    print(adata_unpaired.shape)
    # multivi requires the modalities to be sorted (genes first), this is already the case
    #adata_unpaired = adata_unpaired[:,adata_unpaired.var['feature_types'].argsort()]
    #print(adata_unpaired.var)
    adata_unpaired.write_h5ad('data/'+data_name+'/bonemarrow_trainset_'+str(int(fraction_unpaired*100))+'percent_unpaired.h5ad')
"""