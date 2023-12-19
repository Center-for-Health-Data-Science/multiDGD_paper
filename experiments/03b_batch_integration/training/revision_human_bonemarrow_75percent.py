import mudata as md
import anndata as ad
import pandas as pd
import numpy as np

from omicsdgd import DGD

# create argument parser for the batch left out
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=0)
args = parser.parse_args()
random_seed = args.random_seed
print(
    "training multiDGD on human bonemarrow data with random seed ",
    random_seed
)

###
# load data
###
data_name = "human_bonemarrow"
adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
# create train set
adata.X = adata.layers["counts"]
trainset = adata[adata.obs["train_val_test"] != "test"].copy()
print("length of train indices: ", len(trainset))
# select only 75 percent of train set
np.random.seed(random_seed)
indices_keep = np.random.choice(
    np.arange(len(trainset)), int(0.75 * trainset.n_obs), replace=False
)
train_indices = list(np.where(trainset.obs["train_val_test"] == "train")[0])
print("length of train indices: ", len(train_indices))
train_val_split = [
    train_indices,
    list(np.where(trainset.obs["train_val_test"] == "validation")[0]),
]
print("   data loaded")

###
# initialize model
###
hyperparameters = {
    "latent_dimension": 20,
    "n_hidden": 2,
    "n_hidden_modality": 3,
    "log_wandb": ["viktoriaschuster", "omicsDGD_revision"],
}

model = DGD(
    data=trainset,
    parameter_dictionary=hyperparameters,
    train_validation_split=train_val_split,
    modalities="feature_types",
    meta_label="cell_type",
    correction="Site",
    save_dir="../results/trained_models/" + data_name + "/",
    model_name=data_name + "_l20_h2-3_75percent_rs" + str(random_seed),
    random_seed=random_seed,
)
print("   model initialized")

###
# train and save
###
model.train(n_epochs=1000, train_minimum=100, developer_mode=True, stop_after=50)
model.save()
print("   model saved")

###
# predict for test set
###
original_name = model._model_name
testset = adata[adata.obs["train_val_test"] == "test"].copy()
adata = None
model._model_name = original_name + "_test50e"
model.predict_new(testset, n_epochs=50)
print("   test set inferred (long)")
