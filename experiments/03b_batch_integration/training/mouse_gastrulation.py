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
hyperparameters = {
    "latent_dimension": 20,
    "n_hidden": 2,
    "n_hidden_modality": 2,
    "log_wandb": ["viktoriaschuster", "omicsDGD_revision"],
}

model = DGD(
    data=mudata,
    parameter_dictionary=hyperparameters,
    train_validation_split=train_val_split,
    meta_label="celltype",
    correction="stage",
    save_dir="../results/trained_models/mouse_gastrulation/",
    model_name="mouse_gast_l20_h2-2_rs" + str(random_seed) + "_leftout_" + stages[batch_left_out],
    random_seed=random_seed,
)
print("   model initialized")

###
# train and save
###
model.train(n_epochs=1000, train_minimum=100, developer_mode=True, stop_after=50)
model.save()
print("   model saved")