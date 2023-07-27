import mudata as md
import pandas as pd

from omicsdgd import DGD

# define seed in command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=0)
args = parser.parse_args()
random_seed = args.random_seed

###
# load data
###
data_name = 'human_brain'

mudata = md.read('data/'+data_name+'/mudata.h5mu', backed=False)
mudata.obs['celltype'] = mudata['rna'].obs['atac_celltype']
print(mudata)

# train-validation-test split for reproducibility
# best provided as list [[train_indices], [validation_indices]]
is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
train_val_split = [
    list(is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values),
    list(is_train_df[is_train_df['is_train'] == 'test']['num_idx'].values)
]

###
# initialize model
###
hyperparameters = {
    'latent_dimension': 20,
    'n_hidden': 2,
    'n_hidden_modality': 3,
    'dirichlet_a': 2,
    'log_wandb': ['viktoriaschuster', 'omicsDGD']
}
#random_seed = 8790
#random_seed = 37

model = DGD(data=mudata, 
            parameter_dictionary=hyperparameters,
            train_validation_split=train_val_split,
            meta_label='celltype',
            save_dir='./results/trained_models/'+data_name+'/',
            model_name='human_brain_l20_h2-3_a2_rs'+str(random_seed),
            random_seed=random_seed)
# minimum requirement is DGD(data)
# but adding information about what feature you expect to have clustered and correction factors helps a lot

###
# train and save
###

model.train(n_epochs=1000, train_minimum=100, developer_mode=True, stop_after=10)
# train technically needs no input, but specifying how long it should be trained and what the desired performance metric is helps get the best results

model.save()
print('model saved')

###
# predict for test set
###

from omicsdgd.functions._data_manipulation import load_testdata_as_anndata

save_dir = 'results/trained_models/'

trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
trainset.obs['celltype'] = trainset.obs['atac_celltype']
testset.obs['celltype'] = testset.obs['atac_celltype']
mudata = None

model.predict_new(testset)
print('new samples learned')