import mudata as md
import anndata as ad
import pandas as pd

from omicsdgd import DGD

# create argument parser for the batch left out
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_left_out', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=0)
args = parser.parse_args()
batch_left_out = args.batch_left_out
random_seed = args.random_seed

###
# load data
###
data_name = 'human_bonemarrow'
#batch_left_out = 3

adata = ad.read_h5ad('data/'+data_name+'/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
batches = adata.obs['Site'].unique()

# train-validation-test split for reproducibility
# best provided as list [[train_indices], [validation_indices]]
is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
train_indices_all = list(is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values)
train_indices = [x for x in train_indices_all if adata.obs['Site'].values[x] != batches[batch_left_out]]
train_val_split = [
    train_indices,
    list(is_train_df[is_train_df['is_train'] == 'test']['num_idx'].values)
]

###
# initialize model
###
hyperparameters = {
    'latent_dimension': 20,
    'n_hidden': 2,
    'n_hidden_modality': 3,
    #'n_components': 20,
    'log_wandb': ['viktoriaschuster', 'omicsDGD']
}
#random_seed = 8790
#random_seed = 37
#random_seed = 0

model = DGD(data=adata,
            parameter_dictionary=hyperparameters,
            train_validation_split=train_val_split,
            modalities='feature_types',
            meta_label='cell_type',
            correction='Site',
            save_dir='./results/trained_models/'+data_name+'/',
            model_name=data_name+'_l20_h2-3_leftout_'+batches[batch_left_out],
            random_seed=random_seed)

# minimum requirement is DGD(data)
# but adding information about what feature you expect to have clustered and correction factors helps a lot

###
# train and save
###

model.train(n_epochs=1000, train_minimum=100, developer_mode=True, stop_after=50)
# train technically needs no input, but specifying how long it should be trained and what the desired performance metric is helps get the best results

model.save()
print('model saved')
exit()

###
# predict for test set
###

from omicsdgd.functions._data_manipulation import load_testdata_as_anndata

save_dir = 'results/trained_models/'

trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)

model.predict_new(testset)
print('new samples learned')