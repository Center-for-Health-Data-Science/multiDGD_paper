#data_name = 'human_bonemarrow'
#meta_label='cell_type'
#correction='Site'
data_name = 'mouse_gastrulation'
meta_label='celltype'
correction='stage'
#data_name = 'human_brain'
#meta_label = 'celltype'
#correction = None
reduction_type = 'umap' # options are pca, tsne, umap

#############################
# no changes needed below
#############################

import os
import scvi
import scanpy as sc
import anndata as ad
import mudata as md
import pandas as pd
import numpy as np
import scipy
import scipy.sparse
from sklearn import preprocessing
from sklearn.metrics import adjusted_rand_score, silhouette_score

from omicsdgd.functions._data_manipulation import load_data_from_name
from omicsdgd.functions._analysis import make_palette_from_meta

# create directory for plots (if necessary)
save_dir = 'results/analysis/model_comparison/multiVI/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load data
cluster_class_neworder, class_palette = make_palette_from_meta(data_name)
data = load_data_from_name(data_name)
if type(data) is md.MuData:
    mudata = data
    modality_switch = mudata['rna'].X.shape[1]
    data = ad.AnnData(scipy.sparse.hstack((mudata['rna'].X,mudata['atac'].X)))
    if data_name == 'human_brain':
        data.obs[meta_label] = mudata.obs[meta_label].values
    else:
        data.obs[meta_label] = mudata['rna'].obs[meta_label].values
    if correction is not None:
        data.obs[correction] = mudata['atac'].obs[correction].values
    data.var['feature_type'] = 'ATAC'
    data.var['feature_type'][:modality_switch] = 'GEX'
    mudata = None
else:
    data.X = data.layers['counts'] # they want unnormalized data in X
data.X = data.X.tocsr()
data.obs['modality'] = 'paired'

scvi.settings.seed = 0

model_dir = 'results/trained_models/multiVI/'+data_name+'/'

is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
train_indices = is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values
if not isinstance(data.X, scipy.sparse.csc_matrix):
    data.X = data.X.tocsr()
data = data[train_indices]



# get model names for this dataset
#model_names = [x for x in os.listdir('results/trained_models/multiVI/'+data_name) if '.' not in x]
#model_names = ['l20_e1_d1','l20_e1_d1_rs37','l20_e1_d1_rs8790']
model_names = ['l20_e2_d2','l20_e2_d2_rs37','l20_e2_d2_rs8790']
print(model_names)

data.var_names_make_unique()
data.obs['modality'] = 'paired'
if data_name == 'mouse_gastrulation':
    scvi.model.MULTIVI.setup_anndata(data, batch_key='stage')
elif data_name == 'human_bonemarrow':
    scvi.model.MULTIVI.setup_anndata(data, batch_key='Site')
else:
    scvi.model.MULTIVI.setup_anndata(data)

count = 0
for model_name in model_names:
    print(model_name)

    mvi = scvi.model.MULTIVI.load(model_dir+model_name, data)

    #elbo = mvi.get_elbo()
    #print(f'Elbo for {model_name} is {elbo}')

    n_neighbors = 20
    min_dist = 0.5
    leiden_resolution = 2
    name_attachment = 'n' + str(n_neighbors) + '-m' + str(min_dist) + '-l' + str(leiden_resolution)

    data.obsm['latent'] = mvi.get_latent_representation()
    sc.pp.neighbors(data, use_rep='latent', n_neighbors=n_neighbors)
    sc.tl.leiden(data, key_added='clusters', resolution=leiden_resolution)
    # get the number of clusters from leiden
    n_clusters = len(np.unique(data.obs['clusters'].values))
    print('number of leiden clusters: ', n_clusters)

    le = preprocessing.LabelEncoder()
    le.fit(data.obs[meta_label].values)
    true_labels = le.transform(data.obs[meta_label].values)
    cluster_labels = data.obs['clusters'].values
    radj = adjusted_rand_score(true_labels, np.asarray(cluster_labels))
    print(radj)

    if correction is not None:
        asw = silhouette_score(data.obsm['latent'], data.obs[correction])
        print(asw)

    #sc.tl.umap(data, min_dist=min_dist)
    #sc.pl.umap(data, color=meta_label, save='_'+data_name+'_'+model_name+'_latent_'+meta_label+'_'+name_attachment)
    #if correction is not None:
    #    sc.pl.umap(data, color=correction, save='_'+data_name+'_'+model_name+'_latent_'+correction+'_'+name_attachment)