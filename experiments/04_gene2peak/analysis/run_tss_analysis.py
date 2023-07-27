## Compute perturbation effect as a function of distance to gene TSS ## 

import os,sys
import multiDGD
import numpy as np
import pandas as pd
import anndata as ad
import mudata
import torch
import multiprocessing
import genomic_features as gf
import bioframe

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    default='/nfs/team205/ed6/data/bonemarrow_model/',
                    type=str,
                    help="Directory containing model and data")
args = parser.parse_args()

model_dir = args.model_dir

def get_TSS(gene_ranges_df: pd.DataFrame, perturbed_gene: str):
    '''
    Get position of TSS of gene of interest
    '''
    try:
        gene_range = gene_ranges_df.loc[var_gex.loc[perturbed_gene]['gene_id']]
    except KeyError:
        gene_range = gene_ranges_df.loc[perturbed_gene]
    if gene_range['seq_strand'] > 0:
        tss = gene_range['gene_seq_start']
    else:
        tss = gene_range['gene_seq_end']
    tss_range = gene_range.copy()
    tss_range['gene_seq_start'] = tss
    tss_range['gene_seq_end'] = tss+1
    tss_range['seq_strand'] = 1
    tss_range['gene_name'] = tss_range.name
    
    # to bioframe
    tss_range = pd.DataFrame(tss_range[['seq_name', 'gene_seq_start', 'gene_seq_end', 'gene_name']]).T
    tss_range.columns = ['chrom', 'start', 'end', 'name']
    tss_range['chrom'] = 'chr' + tss_range['chrom']
    return(tss_range.convert_dtypes())

def get_close_peaks_effect(atac_delta_adata: ad.AnnData, gene_ranges_df: pd.DataFrame, perturbed_gene: str) -> None:
    '''
    Get perturbation effect at peaks around TSS
    
    Params:
    -------
    - atac_delta_adata: AnnData of perturbation effects on ATAC modality
    - gene_ranges_df: DataFrame of genomic positions for all genes
    - perturbed_gene: string of name of gene of interest
    '''
    ## Get position of promoter of perturbed gene
    tss_range = get_TSS(gene_ranges_df, perturbed_gene)

    ## Get peaks within 10kb of TSS and distance
    window = 10000
    close_peaks_df = bioframe.closest(peaks_df[peaks_df['chrom'] == tss_range.chrom[0]], tss_range)[['chrom', 'start', 'end', 'name', 'distance']]
    # close_peaks_df = close_peaks_df[close_peaks_df['distance'] < window]
    close_peaks_df = close_peaks_df[close_peaks_df['distance'] < window]

    ## Store mean perturbation delta for close peaks
    close_peaks_df['mean_delta'] =  np.abs(atac_delta_adata[:, close_peaks_df.name].X).mean(0).toarray()
    close_peaks_df['se_delta'] = np.abs(atac_delta_adata[:, close_peaks_df.name].X).std(0).toarray() / np.sqrt(close_peaks_df.shape[0])
    close_peaks_df['perturbed_gene'] = perturbed_gene
    close_peaks_df.to_csv(model_dir + f'TSS_perturb_experiment.{perturbed_gene}.csv')
    
# Read data
data = ad.read_h5ad(model_dir + 'GSE194122_openproblems_neurips2021_multiome_BMMC_processed_withDataSplit.h5ad')
peaks_df = pd.read_csv(model_dir + 'GSE194122_openproblems_neurips2021_multiome_BMMC_processed_withDataSplit.peaks.csv', index_col=0)
peaks_df['name'] = peaks_df[['chrom', 'start', 'end']].astype(str).apply(lambda x: "-".join(x), axis=1)
gene_names = pd.read_csv(model_dir + 'GSE194122_openproblems_neurips2021_multiome_BMMC_processed_withDataSplit.gene_names.csv', index_col=0) 
all_genes = pd.read_table(model_dir + 'genes_to_test.txt', header=None)[0].tolist()

# Load trained model
model = multiDGD.DGD.load(data=data, save_dir=model_dir, model_name='dgd_bonemarrow_default_trained_and_tested')

# specify the samples we want to look at
testset = data[data.obs["train_val_test"] == "test",:].copy()

var_gex = data.var[data.var['modality'] == 'GEX']
var_atac = data.var[data.var['modality'] == 'ATAC']

# Get positions of genes
ensdb = gf.ensembl.annotation(species="Hsapiens", version="108")
gene_ranges_df = ensdb.genes()
try:
    gene_ranges_df = gene_ranges_df[gene_ranges_df['gene_id'].isin(var_gex.gene_id)].copy()
    gene_ranges_df = gene_ranges_df.set_index("gene_id")
except KeyError:
    gene_ranges_df = gene_ranges_df.set_index("gene_name")

## Run gene perturbation and save
all_close_peaks_df = pd.DataFrame()
for g in all_genes:
    if not os.path.exists(model_dir + f'TSS_perturb_experiment.{g}.csv'):
        predicted_changes, samples_of_interest = model.gene2peak(gene_name=g, testset=testset)
        delta_gex = predicted_changes[0]
        delta_atac = predicted_changes[1]

        atac_delta_adata = ad.AnnData(X = delta_atac.numpy(), obs = testset[samples_of_interest].obs, var=var_atac)
        get_close_peaks_effect(atac_delta_adata, gene_ranges_df, g)