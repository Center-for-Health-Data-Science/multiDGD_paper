# check if there is a subfolder called data
if [ ! -d "data" ]; then
    mkdir data
fi
# check if data/raw exists
if [ ! -d "data/raw" ]; then
    mkdir data/raw
fi

# download human bonemarrow data
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
mv ./GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz ./data/raw/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
# unzip
gunzip data/raw/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
# delete the zipped file
rm ./data/raw/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz

# download mouse gastrulation data
### ask Emma, I got the processed data

# download human brain data from https://github.com/GreenleafLab/brainchromatin/blob/main/links.txt
wget https://atrev.s3.amazonaws.com/brainchromatin/hft_ctx_w21_dc1r3_r1/outs/filtered_feature_bc_matrix.h5
mv ./filtered_feature_bc_matrix.h5 data/raw/brain_filtered_feature_bc_matrix.h5
# download annotations
wget https://atrev.s3.amazonaws.com/brainchromatin/multiome_cell_metadata.txt
wget https://atrev.s3.amazonaws.com/brainchromatin/multiome_cluster_names.txt
mv multiome_cell_metadata.txt data/raw/brain_multiome_cell_metadata.txt
mv multiome_cluster_names.txt data/raw/brain_multiome_cluster_names.txt