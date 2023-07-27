import mudata as md
import anndata as ad
import pandas as pd

from omicsdgd import DGD

# create argument parser for random seed and fraction
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--fraction', type=float)
args = parser.parse_args()
random_seed = args.random_seed
fraction = args.fraction

#fraction_options = [0.01, 0.1, 0.25, 0.5, 0.75]

###
# load data
###
data_name = 'human_bonemarrow'

adata = ad.read_h5ad('data/'+data_name+'/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')

# get subset of train indices
df_subset_ids = pd.read_csv('data/'+data_name+'/data_subsets.csv')
train_indices = list(df_subset_ids[(df_subset_ids['fraction'] == fraction) & (df_subset_ids['include'] == 1)]['sample_idx'].values)
print(len(train_indices))
n_samples = len(train_indices)

# train-validation-test split for reproducibility
# best provided as list [[train_indices], [validation_indices]]
is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
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
            model_name=data_name+'_l20_h2-3_rs'+str(random_seed)+'_subset'+str(n_samples),
            random_seed=random_seed)
print(model.representation.z.shape)
print(model.correction_rep.z.shape)

# minimum requirement is DGD(data)
# but adding information about what feature you expect to have clustered and correction factors helps a lot

###
# train and save
###

model.train(n_epochs=1000, train_minimum=100, developer_mode=True, stop_after=20)
# train technically needs no input, but specifying how long it should be trained and what the desired performance metric is helps get the best results

model.save()
print('model saved')

test_indices = list(is_train_df[is_train_df['is_train'] == 'iid_holdout']['num_idx'].values)
testset = adata[test_indices, :].copy()
adata = None
model.predict_new(testset, n_epochs=50)
print('new samples learned')