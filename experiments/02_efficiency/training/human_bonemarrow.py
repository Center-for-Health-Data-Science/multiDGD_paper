import mudata as md
import anndata as ad
import pandas as pd
import numpy as np

from omicsdgd import DGD

# create argument parser for random seed and fraction
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--fraction", type=float)
args = parser.parse_args()
random_seed = args.random_seed
fraction = args.fraction
print(
    "training multiDGD on human bonemarrow data with random seed ",
    random_seed,
    " and fraction ",
    fraction,
)

###
# load data
###
data_name = "human_bonemarrow"
adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
adata.X = adata.layers["counts"] # I seem to have to do it again

# get subset of train indices
df_subset_ids = pd.read_csv('../../data/'+data_name+'_data_subsets.csv')
train_indices = list(df_subset_ids[(df_subset_ids['fraction'] == fraction) & (df_subset_ids['include'] == 1)]['sample_idx'].values)
print(len(train_indices))
n_samples = len(train_indices)

# train-validation-test split for reproducibility
# best provided as list [[train_indices], [validation_indices]]
train_val_split = [
    train_indices,
    list(np.where(adata.obs["train_val_test"] == "validation")[0]),
]
n_samples = len(train_val_split[0])

###
# initialize model
###
hyperparameters = {
    "latent_dimension": 20,
    "n_hidden": 2,
    "n_hidden_modality": 3,
    "log_wandb": ["viktoriaschuster", "omicsDGD"],
}

model = DGD(
    data=adata,
    parameter_dictionary=hyperparameters,
    train_validation_split=train_val_split,
    modalities="feature_types",
    meta_label="cell_type",
    correction="Site",
    save_dir="../results/trained_models/" + data_name + "/",
    model_name=data_name
    + "_l20_h2-3_rs"
    + str(random_seed)
    + "_subset"
    + str(n_samples),
    random_seed=random_seed,
)

###
# train and save
###
model.train(n_epochs=1000, train_minimum=100, developer_mode=True, stop_after=20)
model.save()
print("   model saved")

test_indices = list(np.where(adata.obs["train_val_test"] == "test")[0])
testset = adata[test_indices, :].copy()
adata = None
model.predict_new(testset, n_epochs=50)
print("   new samples learned")
