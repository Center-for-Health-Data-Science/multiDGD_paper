import pandas as pd
import anndata as ad
from omicsdgd import DGD
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scvi
from omicsdgd.functions._data_manipulation import load_testdata_as_anndata
from omicsdgd.functions._analysis import make_palette_from_meta

save_dir = 'results/trained_models/'
data_name = 'human_bonemarrow'
model_name = 'human_bonemarrow_l20_h2-3_rs0_unpaired10percent'
fraction_unpaired = 0.1
#fraction_unpaired_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
train_indices = is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values
df_unpaired = pd.read_csv('data/'+data_name+'/unpairing.csv')
adata = ad.read_h5ad('data/'+data_name+'/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
# make test set now to be able to free memory earlier
adata_test = adata[is_train_df[is_train_df['is_train'] == 'iid_holdout']['num_idx'].values,:].copy()
print('loaded data')

if fraction_unpaired == 0.0:
    model = DGD.load(data=adata[train_indices, :], save_dir=save_dir + data_name + "/", model_name=model_name)
    adata = None
else:
    mod_1_indices = df_unpaired[
        (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "rna")
    ]["sample_idx"].values
    mod_2_indices = df_unpaired[
        (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "atac")
    ]["sample_idx"].values
    if fraction_unpaired < 1.:
        # train-validation-test split for reproducibility
        remaining_indices = df_unpaired[
            (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "paired")
        ]["sample_idx"].values
        print("made indices")
        adata_rna = adata[mod_1_indices, adata.var["feature_types"] == "GEX"].copy()
        print("copied rna")
        adata_atac = adata[mod_2_indices, adata.var["feature_types"] == "ATAC"].copy()
        print("copied atac")
        adata_multi = adata[remaining_indices, :].copy()
        print("copied rest")
        adata_unpaired = scvi.data.organize_multiome_anndatas(adata_multi, adata_rna, adata_atac)
        print("organized data")
        adata_rna, adata_atac, adata_multi = None, None, None

        model = DGD.load(data=adata_unpaired, save_dir=save_dir + data_name + "/", model_name=model_name)
        adata_unpaired = None
    else:
        adata_unpaired = adata[mod_1_indices,:].copy()
        adata_unpaired.obs['modality'] = 'GEX'
        adata_temp = adata[mod_2_indices,:].copy()
        adata_temp.obs['modality'] = 'ATAC'
        adata_unpaired = adata_unpaired.concatenate(adata_temp)
        print("organized data")
        adata, adata_temp = None, None
        model = DGD.load(data=adata_unpaired, save_dir=save_dir + data_name + "/", model_name=model_name)
        adata_unpaired = None
print("loaded model")

# get test set
adata_rna = adata_test[:, adata_test.var['feature_types'] == 'GEX'].copy()
print('copied test rna')
adata_atac = adata_test[:, adata_test.var['feature_types'] == 'ATAC'].copy()
print('copied test atac')
adata_unpaired_test = scvi.data.organize_multiome_anndatas(adata_test, adata_rna, adata_atac)
print(adata_unpaired_test)
print('organized test data')
adata_rna, adata_atac = None, None


rep = model.test_rep.z.detach().numpy()
cell_labels = adata_unpaired_test.obs['cell_type'].values
batch_labels = adata_unpaired_test.obs['Site'].values
modality_labels = adata_unpaired_test.obs['modality'].values
cell_indices = [x.split('_')[0] for x in adata_unpaired_test.obs_names.values]

rep_df = pd.DataFrame(rep, columns=np.arange(rep.shape[1]))
rep_df['cell_type'] = cell_labels
rep_df['batch'] = batch_labels
rep_df['modality'] = modality_labels
rep_df['cell_idx'] = cell_indices
rep_df.to_csv('results/analysis/modality_integration/'+model_name+'_testlatent.csv')