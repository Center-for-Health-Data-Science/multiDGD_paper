'''
I am a great python programmer

This script quantifies the quality of the feature prediction of the DGD model
by computing the reconstruction error of the test set and
by computing the KL divergence between the predicted and the true feature distributions
'''

# imports
import pandas as pd
import numpy as np
import anndata as ad
import torch

from omicsdgd import DGD
from omicsdgd.functions._data_manipulation import load_testdata_as_anndata

####################
# collect test errors per model and sample
####################
# load data and model
save_dir = 'results/trained_models/'
data_name = 'human_bonemarrow'
# get feature names
adata = ad.read_h5ad('data/'+data_name+'/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
rna_features = adata.var[adata.var['feature_types'] == 'GEX'].index
atac_features = adata.var[adata.var['feature_types'] == 'ATAC'].index
adata = None
trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
nonzeros_train = np.count_nonzero(trainset.X.toarray(), axis=0)

model_name = 'human_bonemarrow_l20_h2-3'
model = DGD.load(data=trainset,
    save_dir=save_dir+data_name+'/',
    model_name=model_name)
trainset = None # free memory
model.init_test_set(testset)

# get test predictions
print('predicting test samples')
test_predictions = model.predict_from_representation(model.test_rep, model.correction_test_rep)
# get test errors
print('computing test errors')
test_errors = model.get_prediction_errors(test_predictions, model.test_set, reduction='gene')
test_errors = test_errors.sum(axis=0).detach().numpy()

# get rescaled predictions
print('getting rescaled predictions')
test_reconstructions = [test_predictions[0]*model.test_set.library[:,0].unsqueeze(1), test_predictions[1]*model.test_set.library[:,1].unsqueeze(1)]
test_predictions = None # free memory

# get kullback leibler divergence
def compute_distribution_frequencies(a, b):
    '''
    calculate the frequencies of observations in two discrete distributions a and b
    '''
    a = np.round(a.numpy())
    b = np.round(b.numpy())
    unique_values = np.sort(np.unique(np.concatenate([a,b])))
    a_frequencies = np.array([np.sum(a==value) for value in unique_values])
    b_frequencies = np.array([np.sum(b==value) for value in unique_values])
    a_frequencies = a_frequencies / np.sum(a_frequencies)
    b_frequencies = b_frequencies / np.sum(b_frequencies)
    return torch.Tensor(a_frequencies), torch.Tensor(b_frequencies)

def discrete_kullback_leibler(p, q):
    '''
    Compute the discrete Kullback-Leibler divergence between two distributions.
    It can be seen as the information gain 
    achieved by using the real distribution p instead of the approximate distribution q.
    '''
    p = p + 1e-10
    q = q + 1e-10
    return (p * torch.log(p / q)).sum().item()

def compute_featurewise_kullback_leibler_divergence(x, y):
    '''
    compute the kullback leibler divergence between two discrete distributions
    which are represented by data features x and y
    x is the real output and y is the predicted output
    '''
    dkl = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        a = x[:,i]
        b = y[:,i]
        freq_a, freq_b = compute_distribution_frequencies(a, b)
        dkl[i] = discrete_kullback_leibler(freq_a, freq_b)
    return dkl

print('calculating Kullback-Leibler divergences')
rna_dkl = compute_featurewise_kullback_leibler_divergence(model.test_set.data[:,:model.train_set.modality_switch], test_reconstructions[0].detach())
atac_dkl = compute_featurewise_kullback_leibler_divergence(model.test_set.data[:,model.train_set.modality_switch:], test_reconstructions[1].detach())

# save information in dataframes and export to csv
df_rna = pd.DataFrame({
    'feature_name':rna_features,
    'reconstruction_error':test_errors[:model.train_set.modality_switch],
    'kullback_leibler_divergence':rna_dkl,
    'non_zero_counts_train': nonzeros_train[:model.train_set.modality_switch]
})
df_atac = pd.DataFrame({
    'feature_name':atac_features,
    'reconstruction_error':test_errors[model.train_set.modality_switch:],
    'kullback_leibler_divergence':atac_dkl,
    'non_zero_counts_train': nonzeros_train[model.train_set.modality_switch:]
})
df_out = pd.concat([df_rna, df_atac])
df_out.to_csv('results/analysis/gene_to_peak/'+model_name+'_feature_errors_and_dkl.csv', index='feature_name')