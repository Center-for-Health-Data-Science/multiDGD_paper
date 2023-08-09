############################
# preparing the human brain data
# the result is the mudata object on figshare
############################
print("preparing the human brain data")

###
# h5 processing code
# from https://support.10xgenomics.com/single-cell-multiome-atac-gex/software/pipelines/latest/advanced/h5_matrices
###
import collections
import scipy.sparse as sp_sparse
import tables
import pandas as pd
import anndata as ad
import numpy as np
import mudata as md

CountMatrix = collections.namedtuple(
    "CountMatrix", ["feature_ref", "barcodes", "matrix"]
)


def get_matrix_from_multiple_h5s(filenames):
    for i, filename in enumerate(filenames):
        with tables.open_file(filename, "r") as f:
            mat_group = f.get_node(f.root, "matrix")
            barcodes = [
                x.decode("UTF-8") for x in f.get_node(mat_group, "barcodes").read()
            ]
            data = getattr(mat_group, "data").read()
            indices = getattr(mat_group, "indices").read()
            indptr = getattr(mat_group, "indptr").read()
            shape = getattr(mat_group, "shape").read()
            matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)

            feature_group = f.get_node(mat_group, "features")
            feature_ids = [
                x.decode("UTF-8") for x in getattr(feature_group, "id").read()
            ]
            feature_names = [
                x.decode("UTF-8") for x in getattr(feature_group, "name").read()
            ]
            feature_types = [
                x.decode("UTF-8") for x in getattr(feature_group, "feature_type").read()
            ]
            feature_ref = pd.DataFrame(
                {
                    "id": feature_ids,
                    "name": feature_names,
                    "feature_types": feature_types,
                }
            )
            barcode_ref = pd.DataFrame({"barcodes": barcodes})
        if i == 0:
            out_matrix = matrix
            out_feature_df = feature_ref
            out_barcode_df = barcode_ref
            return [
                out_matrix,
                out_feature_df,
                out_barcode_df,
            ]  # because the other 2 seem to be repeats
        else:
            out_matrix = sp_sparse.hstack([out_matrix, matrix])
            pd.testing.assert_frame_equal(feature_ref, out_feature_df)
            out_barcode_df = out_barcode_df.append(barcode_ref)

    return [out_matrix, out_feature_df, out_barcode_df]


# meta data taken from https://github.com/GreenleafLab/brainchromatin/blob/main/links.txt
print("    collecting data and meta data")
cell_meta_data = pd.read_csv("../../data/raw/brain_multiome_cell_metadata.txt", sep="\t")
cluster_names = pd.read_csv("../../data/raw/brain_multiome_cluster_names.txt", sep="\t")
cell_meta_data["rna_celltype"] = [
    cluster_names[cluster_names["Assay"] == "Multiome RNA"]["Cluster.Name"].values[
        np.where(
            cluster_names[cluster_names["Assay"] == "Multiome RNA"]["Cluster.ID"].values
            == x
        )[0]
    ][0]
    for x in cell_meta_data["seurat_clusters"].values
]
cell_meta_data["atac_celltype"] = [
    cluster_names[cluster_names["Assay"] == "Multiome ATAC"]["Cluster.Name"].values[
        np.where(
            cluster_names[cluster_names["Assay"] == "Multiome ATAC"][
                "Cluster.ID"
            ].values
            == x
        )[0]
    ][0]
    for x in cell_meta_data["ATAC_cluster"].values
]

filtered_feature_bc_matrix = get_matrix_from_multiple_h5s(
    ["../../data/raw/brain_filtered_feature_bc_matrix.h5"]
)

###
# creating anndata object from the h5 file
###
print("   creating anndata object")
adata = ad.AnnData(filtered_feature_bc_matrix[0].transpose())
adata.obs["ID"] = filtered_feature_bc_matrix[2].values
adata.obs_names = adata.obs["ID"].values
adata.var_names = filtered_feature_bc_matrix[1]["id"].values
adata.var["name"] = filtered_feature_bc_matrix[1]["name"].values
adata.var["modality"] = filtered_feature_bc_matrix[1]["feature_types"].values
# make sure that we only use annotated cells
keep_annotated = [
    x
    for x in range(adata.shape[0])
    if adata.obs["ID"].values[x].split("-")[0] in cell_meta_data["Cell.Barcode"].values
]
adata = adata[keep_annotated, :]
adata.obs["rna_celltype"] = [
    cell_meta_data[cell_meta_data["Cell.Barcode"] == x.split("-")[0]][
        "rna_celltype"
    ].values[0]
    for x in adata.obs["ID"].values
]
adata.obs["atac_celltype"] = [
    cell_meta_data[cell_meta_data["Cell.Barcode"] == x.split("-")[0]][
        "atac_celltype"
    ].values[0]
    for x in adata.obs["ID"].values
]

###
# split gex and atac for feature selection and mudata creation
###
modality_switch = np.where(adata.var["modality"] == "Peaks")[0][0]
adata_gex = adata[:, :modality_switch]
adata_atac = adata[:, modality_switch:]
###

###
# feature selection
###
print("   feature selection")
threshold = 0.01
percent_threshold = int(threshold * adata.shape[0])
gene_nonzero_id, gene_nonzero_count = np.unique(
    adata_gex.X.copy().tocsr().nonzero()[1], return_counts=True
)
selected_features = gene_nonzero_id[
    np.where(gene_nonzero_count >= percent_threshold)[0]
]
modality_switch = len(selected_features)
print("      selected " + str(len(selected_features)) + " gex features")
threshold = 0.01
percent_threshold = int(threshold * adata.shape[0])
atac_nonzero_id, atac_nonzero_count = np.unique(
    adata_atac.X.copy().tocsr().nonzero()[1], return_counts=True
)
selected_features_atac = atac_nonzero_id[
    np.where(atac_nonzero_count >= percent_threshold)[0]
]
print("      selected " + str(len(selected_features_atac)) + " atac features")

###
# make mudata object
###
print("   creating mudata object")
import mudata as md

mudata = md.MuData(
    {
        "rna": adata_gex[:, selected_features],
        "atac": adata_atac[:, selected_features_atac],
    }
)
mudata.obs["celltype"] = adata.obs["atac_celltype"]
mudata.obs["observable"] = mudata.obs["celltype"].values

###
# make train-val-test split
###
dataset_ids = np.arange(adata.shape[0])
np.random.shuffle(dataset_ids)
n_testsamples = int(0.1 * adata.shape[0])
train_val_test_split = pd.DataFrame(
    {"num_idx": np.arange(adata.shape[0]), "is_train": "train"}
)
train_val_test_split["is_train"].values[dataset_ids[:n_testsamples]] = "validation"
train_val_test_split["is_train"].values[
    dataset_ids[n_testsamples : (2 * n_testsamples)]
] = "test"
mudata.obs["train_val_test"] = train_val_test_split["is_train"].values

###
# save data
###
mudata.write("../../data/human_brain.h5mu")
print("   saved data")
