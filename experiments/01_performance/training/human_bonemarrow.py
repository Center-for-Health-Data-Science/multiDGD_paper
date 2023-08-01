import anndata as ad
import pandas as pd
import numpy as np
from omicsdgd import DGD

# define seed in command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=0)
args = parser.parse_args()
random_seed = args.random_seed
print("training multiDGD on human bonemarrow data with random seed ", random_seed)

###
# load data
###
data_name = "human_bonemarrow"
adata = ad.read_h5ad("data/" + data_name + ".h5ad")

# train-validation-test split for reproducibility
# best provided as list [[train_indices], [validation_indices]]
train_val_split = [
    list(np.where(adata.obs["train_val_test"] == "train")[0]),
    list(np.where(adata.obs["train_val_test"] == "validation")[0]),
]

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
    save_dir="./results/trained_models/" + data_name + "/",
    model_name=data_name + "_l20_h2-3_rs" + str(random_seed) + "_sigmoid",
    random_seed=random_seed,
)


###
# train and save
###
print("   training")
model.train(n_epochs=1000, train_minimum=100, developer_mode=True, stop_after=20)
model.save()
print("   model saved")

###
# predict for test set
###
testset = adata[adata.obs["train_val_test"] == "test"].copy()
model.predict_new(testset)
print("   test set inferred")
