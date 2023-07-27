import torch
import numpy as np

from omicsdgd import DGD
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

################################
import anndata as ad
import numpy as np
import mudata as md
import pandas as pd
import scipy
import torch
gex = ad.read_h5ad('data/mouse_gastrulation/raw/anndata.h5ad')
atac = ad.read_h5ad('data/mouse_gastrulation/raw/PeakMatrix_anndata.h5ad')
ids_shared = list(set(gex.obs['sample'].index.values).intersection(set(atac.obs['sample'].index.values)))
ids_gex = np.where(gex.obs['sample'].index.isin(ids_shared))[0]
ids_atac = np.where(atac.obs['sample'].index.isin(ids_shared))[0]
gex = gex[ids_gex]
atac = atac[ids_atac]
threshold = 0.00
if threshold > 0.000:
    percent_threshold = int(threshold * len(ids_shared))
    gene_nonzero_id, gene_nonzero_count = np.unique(gex.X.copy().tocsr().nonzero()[1], return_counts=True)
    selected_features = gene_nonzero_id[np.where(gene_nonzero_count >= percent_threshold)[0]]
    modality_switch = len(selected_features)
    print('selected '+str(len(selected_features))+' gex features')
    percent_threshold = int(threshold * len(ids_shared))
    atac_nonzero_id, atac_nonzero_count = np.unique(atac.X.copy().tocsr().nonzero()[1], return_counts=True)
    selected_features_atac = atac_nonzero_id[np.where(atac_nonzero_count >= percent_threshold)[0]]
    print('selected '+str(len(selected_features_atac))+' atac features')
    mudata = md.MuData({'rna': gex[:,selected_features], 'atac': atac[:,selected_features_atac]})
else:
    mudata = md.MuData({'rna': gex, 'atac': atac})
mudata.obs['stage'] = mudata['atac'].obs['stage']#.values
mudata.obs['celltype'] = mudata['rna'].obs['celltype']#.values
mudata.obs = mudata['rna'].obs
mudata.obs["modality"] = "paired"
modality_switch = mudata['rna'].X.shape[1]
mudata.var = pd.DataFrame(index=mudata['rna'].var_names.tolist()+mudata['atac'].var_names.tolist(),
    data={'name': mudata['rna'].var['gene'].values.tolist()+mudata['atac'].var['idx'].values.tolist(),
    'feature_types': ['rna']*modality_switch+['atac']*(mudata.shape[1]-modality_switch)})

is_train_df = pd.read_csv('data/mouse_gastrulation/train_val_test_split.csv')
train_val_split = [
    list(is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values),
    list(is_train_df[is_train_df['is_train'] == 'test']['num_idx'].values)
]
trainset = mudata[train_val_split[0]].copy()
testset = mudata[list(is_train_df[is_train_df['is_train'] == 'iid_holdout']['num_idx'].values)].copy()
library = torch.cat(
    (torch.tensor(np.asarray(testset['rna'].X.sum(-1))),
    torch.tensor(np.asarray(testset['atac'].X.sum(-1)))),
    dim=1
)
testset_obs = testset.obs.copy()
testset_ad = ad.AnnData(scipy.sparse.hstack((testset['rna'].X,testset['atac'].X)))
testset_ad.obs['stage'] = testset['rna'].obs['stage']#.values
testset_ad.obs['celltype'] = testset['rna'].obs['celltype']#.values
testset_ad.obs["modality"] = "paired"
testset_ad.var = pd.DataFrame(index=testset['rna'].var_names.tolist()+testset['atac'].var_names.tolist(),
    data={'name': testset['rna'].var['gene'].values.tolist()+testset['atac'].var['idx'].values.tolist(),
    'feature_types': ['rna']*modality_switch+['atac']*(testset.shape[1]-modality_switch)})
testset = testset_ad
print(trainset)
print(testset)

random_seed = 0
data_name = 'mouse_gastrulation'
save_dir = 'results/trained_models/'
dgd_name = 'mouse_gast_l20_h2-2_rs'+str(random_seed)+'_scale5'+'_featselect'+str(threshold).split('.')[1]

################################



################################

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

exit()
#model.predict_new(testset, n_epochs=50)
#print('new samples learned')

import matplotlib.pyplot as plt
# plot the first three cells

fig, ax = plt.subplots(1,3, figsize=(10,5))
for i in range(3):
    ax[i].scatter(test_gex[i,:], test_recon_gex[i,:]*test_gex[i,:].sum(), s=1)
plt.savefig('results/analysis/performance_evaluation/reconstruction/'+dgd_name+'_test_recon_gex_2.png')