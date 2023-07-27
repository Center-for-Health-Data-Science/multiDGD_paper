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
#"""
# test different feature selection thresholds
import anndata as ad
import numpy as np
gex = ad.read_h5ad('data/mouse_gastrulation/raw/anndata.h5ad')
atac = ad.read_h5ad('data/mouse_gastrulation/raw/PeakMatrix_anndata.h5ad')
ids_shared = list(set(gex.obs['sample'].index.values).intersection(set(atac.obs['sample'].index.values)))
ids_gex = np.where(gex.obs['sample'].index.isin(ids_shared))[0]
ids_atac = np.where(atac.obs['sample'].index.isin(ids_shared))[0]
gex = gex[ids_gex]
atac = atac[ids_atac]
threshold = 0.00
mudata = md.MuData({'rna': gex, 'atac': atac})
#"""

#mudata = md.read('data/mouse_gastrulation/extras/mudata_05rna1atac_percentCells.h5mu', backed=False)
#mudata = md.read('data/mouse_gastrulation/mudata.h5mu', backed=False)
mudata.obs['stage'] = mudata['atac'].obs['stage']#.values
mudata.obs['celltype'] = mudata['rna'].obs['celltype']#.values

# train-validation-test split for reproducibility
# best provided as list [[train_indices], [validation_indices]]
is_train_df = pd.read_csv('data/mouse_gastrulation/train_val_test_split.csv')
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
    'n_hidden_modality': 2,
    #'softball_scale': 2,
    #'sd_mean': 0.05,
    'dirichlet_a': 1,
    #'n_components': 25,
    #'decoder_width': 2,
    #'batch_size': 64,
    'log_wandb': ['viktoriaschuster', 'omicsDGD']
}
#random_seed = 8790
#random_seed = 37
#random_seed = 0

model = DGD(data=mudata,
            parameter_dictionary=hyperparameters,
            train_validation_split=train_val_split,
            meta_label='celltype',
            correction='stage',
            save_dir='./results/trained_models/mouse_gastrulation/',
            #model_name='mouse_gast_l20_h2-2_c25_rs'+str(random_seed),
            model_name='mouse_gast_l20_h2-2_a1_rs'+str(random_seed),
            #model_name='mouse_gast_l20_h3-3w2_rs'+str(random_seed)+'_scale5'+'_featselect'+str(threshold).split('.')[1],
            random_seed=random_seed)

"""
means = model.gmm.mean.detach().numpy()
samples = model.gmm.sample(10000).detach().numpy()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(means)
means_pca = pca.transform(means)
samples_pca = pca.transform(samples)
df_pca = pd.DataFrame(means_pca, columns=['PC1', 'PC2'])
df_pca['type'] = 'mean'
df_pca_temp = pd.DataFrame(samples_pca, columns=['PC1', 'PC2'])
df_pca_temp['type'] = 'sample'
df_pca = pd.concat([df_pca, df_pca_temp], axis=0)
df_pca['type'] = df_pca['type'].astype('category')
df_pca['type'].cat.set_categories(['sample','mean'], inplace=True)
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data=df_pca.sort_values(by='type'), x='PC1', y='PC2', hue='type', size='type', sizes=(20, 1))
plt.show()
"""

# minimum requirement is DGD(data)
# but adding information about what feature you expect to have clustered and correction factors helps a lot

###
# train and save
###

model.train(n_epochs=1000, train_minimum=100, developer_mode=True, stop_after=20)
# train technically needs no input, but specifying how long it should be trained and what the desired performance metric is helps get the best results

model.save()
print('model saved')

###
# predict for test set
###

from omicsdgd.functions._data_manipulation import load_testdata_as_anndata

save_dir = 'results/trained_models/'

data_name = 'mouse_gastrulation'
trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
mudata = None

model.predict_new(testset)
print('new samples learned')