import torch
import numpy as np

from omicsdgd import DGD

from omicsdgd.functions._data_manipulation import load_testdata_as_anndata
from omicsdgd.functions import set_random_seed

# create argument parser for random seed
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=0)
args = parser.parse_args()
random_seed = args.random_seed
set_random_seed(random_seed)

save_dir = 'results/trained_models/'

data_name = 'mouse_gastrulation'
#if random_seed == 0:
#    dgd_name = 'mouse_gast_l20_h2-2_c20_new2'
#else:
#    dgd_name = 'mouse_gast_l20_h2-2_a2_rs'+str(random_seed)
dgd_name = 'mouse_gast_l20_h2-2_rs'+str(random_seed)
trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
print(np.array(testset.X[0,:20].todense()))
model = DGD.load(data=trainset,
            save_dir=save_dir+data_name+'/',
            model_name=dgd_name)

#"""
# first check if there is a test_rep
print('test_rep: ', model.test_rep)

model.predict_new(testset, n_epochs=50)
print('new samples learned')

test_gex = np.array(testset.X[:,:modality_switch].todense())
# extract test reconstructions
test_recon = model.decoder_forward(model.test_rep.z.shape[0])
# save to numpy
test_recon_gex = test_recon[0].cpu().detach().numpy()
test_recon_atac = test_recon[1].cpu().detach().numpy()
np.save('results/analysis/performance_evaluation/reconstruction/'+dgd_name+'_test_recon_gex.npy', test_recon_gex)
np.save('results/analysis/performance_evaluation/reconstruction/'+dgd_name+'_test_recon_atac.npy', test_recon_atac)
print('saved reconstructions')
# also save the test data if we are at it
#np.save('results/analysis/performance_evaluation/reconstruction/test_counts_gex.npy', test_gex)
#np.save('results/analysis/performance_evaluation/reconstruction/test_counts_atac.npy', np.array(testset.X[:,modality_switch:].todense()))
#print('saved test data')
#"""

#exit()
#model.predict_new(testset, n_epochs=50)
#print('new samples learned')

import matplotlib.pyplot as plt
# plot the first three cells

fig, ax = plt.subplots(1,3, figsize=(10,5))
for i in range(3):
    ax[i].scatter(test_gex[i,:], test_recon_gex[i,:]*test_gex[i,:].sum(), s=1)
plt.savefig('results/analysis/performance_evaluation/reconstruction/'+dgd_name+'_test_recon_gex_2.png')