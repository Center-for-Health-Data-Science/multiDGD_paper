import anndata as ad
import numpy as np

from omicsdgd import DGD
from omicsdgd.functions._data_manipulation import load_testdata_as_anndata

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
#random_seed = 0
save_dir = 'results/trained_models/'
data_name = 'human_bonemarrow'
#batch_left_out = 0

adata = ad.read_h5ad('data/'+data_name+'/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
batches = adata.obs['Site'].unique()

dgd_name = data_name+'_l20_h2-3_leftout_'+batches[batch_left_out]
#dgd_name = data_name+'_l20_h2-3_test2'
# test2 has a learning rate of 0.1 for the correction representation

trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
trainset = trainset[trainset.obs['Site'] != batches[batch_left_out]]

model = DGD.load(data=trainset,
            save_dir=save_dir+data_name+'/',
            model_name=dgd_name)
print('loaded model')

model.predict_new(testset, n_epochs=50, indices_of_new_distribution=np.where(testset.obs['Site'] == batches[batch_left_out])[0])
print('new samples learned')