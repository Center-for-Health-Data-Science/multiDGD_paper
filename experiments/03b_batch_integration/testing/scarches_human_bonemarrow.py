'''
scArches has no support yet for MultiVI,
si I do this for scVI with only expression data

load a trained multiVI (currently scVI) model and integrate a new 'batch' of samples by using scArches
'''
import scvi
import anndata as ad
import pandas as pd
#import scarches as sca

random_seed = 0
scvi.settings.seed = random_seed
batch_left_out = 3

################
# load and train and heldout data
################
print('load data')
# get data
adata = ad.read_h5ad('data/human_bonemarrow/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
# select train data without batch to be left out
is_train_df = pd.read_csv('data/human_bonemarrow/train_val_test_split.csv')
batches = adata.obs['Site'].unique()
train_indices_all = list(is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values)
train_indices = [x for x in train_indices_all if adata.obs['Site'].values[x] != batches[batch_left_out]]
train_indices_scarches = [x for x in train_indices_all if adata.obs['Site'].values[x] == batches[batch_left_out]]
# prepare data for multiVI
adata.var_names_make_unique()
adata.obs['modality'] = 'paired'
adata.X = adata.layers['counts'] # they want unnormalized data in X
adata_train = adata.copy()[train_indices]
scvi.model.MULTIVI.setup_anndata(adata_train, batch_key='Site')

################
# load trained model
################
print('load model')
model_dir = 'results/trained_models/multiVI/human_pbmc/'
model_name = 'l20_e2_d2_leftout_'+batches[batch_left_out]
#mvi = scvi.model.MULTIVI.load(model_dir+model_name, adata_train)

################
# apply scArches
################
print('apply scArches')
# see tutorial for more https://scarches.readthedocs.io/en/latest/scvi_surgery_pipeline.html# 
adata_test = adata.copy()[train_indices_scarches]
scvi.model.MULTIVI.prepare_query_anndata(adata_test, model_dir+model_name)
print('prepared query anndata')
model = scvi.model.MULTIVI.load_query_data(
    adata_test,
    model_dir+model_name
)
print('initialized new model')
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_index = device.index
model.train(max_epochs=200, plan_kwargs=dict(weight_decay=0.0),use_gpu=device_index)

# save new model
model.save(model_dir+model_name+'_scarches')