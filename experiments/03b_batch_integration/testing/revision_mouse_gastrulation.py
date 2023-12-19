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
    "training multiDGD on mouse gastrulation data with random seed ",
    random_seed,
    " and stage left out: ",
    batch_left_out,
)

###
# load data
###
mudata = md.read("../../data/mouse_gastrulation.h5mu", backed=False)
stages = mudata.obs["stage"].unique()
train_indices_all = list(np.where(mudata.obs["train_val_test"] == "train")[0])
train_indices = [
    x
    for x in train_indices_all
    if mudata.obs["stage"].values[x] != stages[batch_left_out]
]
train_val_split = [
    train_indices,
    list(np.where(mudata.obs["train_val_test"] == "validation")[0]),
]
print("   data loaded")

###
# initialize model
###

model = DGD.load(
    data=mudata[train_indices],
    save_dir="../results/trained_models/mouse_gastrulation/",
    model_name="mouse_gast_l20_h2-2_rs" + str(random_seed) + "_leftout_" + stages[batch_left_out],
    random_seed=random_seed,
)
print("   model loaded")

###
# predict for test set
###
original_name = model._model_name
testset = mudata[mudata.obs["train_val_test"] == "test"].copy()

#"""
#mudata = None

model._model_name = original_name + "_test50e_default"
model.predict_new(testset, n_epochs=50)
print("   test set inferred (long)")
#"""

#"""
import torch

# get the covariate components for each covariate of the train set
train_covariate_clusters = torch.argmax(
    model.correction_gmm.sample_probs(model.correction_rep.z), dim=-1
).to(torch.int16)
# check which covariate label is in which cluster
train_batches = mudata[train_indices].obs["stage"].values
batch_id_dict = {}
for i, batch in enumerate(stages):
    if len(train_covariate_clusters[train_batches == batch]) > 0:
        majority_cluster = (
            torch.median(train_covariate_clusters[train_batches == batch])
            .to(torch.int16)
            .item()
        )
        batch_id_dict[batch] = majority_cluster
    else:
        batch_id_dict[batch] = -1
print(batch_id_dict)

# now create a list of the test samples with the correct cluster
init_covariate_supervised = [
    batch_id_dict[batch] for batch in testset.obs["stage"].values
]
indices_of_new_distribution = np.where(np.array(init_covariate_supervised) == -1)[0]
###

mudata = None

#"""
beta = 20
#beta = 5
model._model_name = original_name + "_test100e_covSupervised_beta" + str(beta)
print("testing: ", model._model_name)
model.predict_new(
    testset,
    n_epochs=100,
    indices_of_new_distribution=indices_of_new_distribution,
    start_from_zero=[False, True],
    init_covariate_supervised=init_covariate_supervised,
    train_supervised=True,
    cov_beta=beta
)
print("   test set inferred (test)")