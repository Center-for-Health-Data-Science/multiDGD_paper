import mudata as md
import anndata as ad
import pandas as pd
import numpy as np

from omicsdgd import DGD

# create argument parser for the batch left out
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_left_out", type=int, default=0)
parser.add_argument("--random_seed", type=int, default=0)
args = parser.parse_args()
batch_left_out = args.batch_left_out
random_seed = args.random_seed
print(
    "training multiDGD on human bonemarrow data with random seed ",
    random_seed,
    " and batch left out: ",
    batch_left_out,
)

###
# load data
###
data_name = "human_bonemarrow"
adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
adata.X = adata.layers["counts"]
batches = adata.obs["Site"].unique()
train_indices_all = list(np.where(adata.obs["train_val_test"] == "train")[0])
train_indices = [
    x
    for x in train_indices_all
    if adata.obs["Site"].values[x] != batches[batch_left_out]
]
train_val_split = [
    train_indices,
    list(np.where(adata.obs["train_val_test"] == "validation")[0]),
]
print("   data loaded")

###
# initialize model
###

model = DGD.load(
    data=adata[train_indices],
    save_dir="../results/trained_models/" + data_name + "/",
    model_name=data_name + "_l20_h2-3_leftout_" + batches[batch_left_out],
    random_seed=random_seed
)
print("   model loaded")

###
# predict for test set
###
original_name = model._model_name
model._model_name = original_name + "_test10e_noCovError"
testset = adata[adata.obs["train_val_test"] == "test"].copy()
model.predict_new(testset, include_correction_error=False)
print("   test set inferred")

model._model_name = original_name + "_test50e_noCovError"
model.predict_new(testset, include_correction_error=False, n_epochs=50)
print("   test set inferred (long)")
