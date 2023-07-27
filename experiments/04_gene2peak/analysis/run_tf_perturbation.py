### Run TF perturbation prediction ### 

import os,sys
import multiDGD
import numpy as np
import pandas as pd
import anndata as ad
import mudata
import torch
import multiprocessing
# print(f"CUDA: {torch.cuda.is_available()}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    default='/nfs/team205/ed6/data/bonemarrow_model/',
                    type=str,
                    help="Directory containing model and data")
parser.add_argument("--n_cores",
                    default=10,
                    type=int,
                    help="Number of cores to use for multiprocessing")
parser.add_argument("--n_sample_peaks",
                    default=10000,
                    type=int,
                    help="How many peaks to sample per TF")
args = parser.parse_args()


def get_closing_fraction(perturb_mat):
    closing_fraction = np.mean(np.sign(perturb_mat) < 0, 1).toarray()
    return(closing_fraction)

def run_TF_perturbation(tf_name):
    predicted_changes, samples_of_interest = model.gene2peak(gene_name=tf_name, testset=testset)
    delta_atac = predicted_changes[1]

    atac_delta_adata = ad.AnnData(X = delta_atac.numpy(), obs = testset[samples_of_interest].obs, var=var_atac)

    # Get output (sample same no of peaks for each TF)
    TF_binding_peaks = motifs_in_peaks_clean.index[motifs_in_peaks_clean[tf_name] == 1]
    if len(TF_binding_peaks) > n_sample:
        TF_binding_peaks = np.random.choice(TF_binding_peaks.values.flatten(), n_sample)
    random_peaks = motifs_in_peaks_clean.sample(len(TF_binding_peaks)).index
    
    perturb_res_df = pd.DataFrame({
        'mean_delta_TFmotifs':atac_delta_adata[:,TF_binding_peaks].X.mean(1).toarray(),
        'mean_delta_random':atac_delta_adata[:,random_peaks].X.mean(1).toarray(),
        'frac_closing_TFmotifs':np.mean(atac_delta_adata[:,TF_binding_peaks].X.toarray() < 0, 1),
        'frac_closing_random':np.mean(atac_delta_adata[:,random_peaks].X.toarray() < 0, 1),
        })

    perturb_res_df['TF_name'] = tf_name
    perturb_res_df['TF_mode'] = tf_df[tf_df['Transcription_factor'] == tf_name].GO_term_mode_of_action.values[0]
    
    # Save cell type info
    perturb_res_df['cell_type'] = atac_delta_adata.obs['cell_type'].values.copy()    
    perturb_res_df.to_csv(model_dir + f'TF_perturb_experiment.{tf_name}.csv')

# Parse args
model_dir = args.model_dir
num_processes = args.n_cores
n_sample = args.n_sample_peaks

# Read data
data = ad.read_h5ad(model_dir + 'GSE194122_openproblems_neurips2021_multiome_BMMC_processed_withDataSplit.h5ad')
motifs_in_peaks_clean = pd.read_csv(model_dir + 'GSE194122_openproblems_neurips2021_multiome_BMMC_processed_withDataSplit.motifs_in_peaks.clean.csv', index_col=0)
all_tfs = pd.read_table(model_dir + 'TF_to_test.txt', header=None)[0].tolist()
tf_df = pd.read_table(model_dir + 'TF_mode_domcke_2020.txt')

# Load trained model
model = multiDGD.DGD.load(data=data, save_dir=model_dir, model_name='dgd_bonemarrow_default_trained_and_tested')

# specify the samples we want to look at
testset = data[data.obs["train_val_test"] == "test",:].copy()

var_gex = data.var[data.var['modality'] == 'GEX']
var_atac = data.var[data.var['modality'] == 'ATAC']

for tf in all_tfs:
    if not os.path.exists(model_dir + f'TF_perturb_experiment.{tf}.csv'):
        run_TF_perturbation(tf)