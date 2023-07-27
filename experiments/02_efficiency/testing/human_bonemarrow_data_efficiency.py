import pandas as pd
import anndata as ad

from omicsdgd import DGD

from omicsdgd.functions._data_manipulation import load_testdata_as_anndata
from omicsdgd.functions import set_random_seed

# create argument parser for random seed and fraction
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--fraction', type=float)
args = parser.parse_args()
random_seed = args.random_seed
fraction = args.fraction
set_random_seed(random_seed)

save_dir = 'results/trained_models/'

#random_seed = 8790
#random_seed = 37
#random_seed = 0

#fraction_options = [0.01, 0.1, 0.25, 0.5, 0.75]
#fraction = 0.1

data_name = 'human_bonemarrow'

trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
trainset = None
adata = ad.read_h5ad('data/'+data_name+'/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
# get subset of train indices
df_subset_ids = pd.read_csv('data/'+data_name+'/data_subsets.csv')
train_indices = list(df_subset_ids[(df_subset_ids['fraction'] == fraction) & (df_subset_ids['include'] == 1)]['sample_idx'].values)
n_samples = len(train_indices)
print(n_samples)
trainset = adata[train_indices].copy()

#dgd_name = '_l20_h2-3_rs'+str(random_seed)
dgd_name = data_name+'_l20_h2-3_rs'+str(random_seed)+'_subset'+str(n_samples)
print(dgd_name)


model = DGD.load(data=trainset,
            save_dir=save_dir+data_name+'/',
            model_name=dgd_name)
print('loaded model')

model.predict_new(testset, n_epochs=50)
print('new samples learned')