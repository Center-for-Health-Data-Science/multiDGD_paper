import anndata as ad
import pandas as pd
import numpy as np
from omicsdgd import DGD

# define seed in command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--train_minimum", type=int, default=100)
parser.add_argument("--stop_after", type=int, default=20)
args = parser.parse_args()
random_seed = args.random_seed
epochs = args.epochs
train_minimum = args.train_minimum
stop_after = args.stop_after
print("training multiDGD on human bonemarrow data with random seed ", random_seed)

###
# load data
###
data_name = "human_bonemarrow"
adata = ad.read_h5ad("../../data/" + data_name + ".h5ad")
adata.X = adata.layers["counts"] # I seem to have to do it again

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
    save_dir="../results/trained_models/" + data_name + "/",
    model_name=data_name + "_l20_h2-3_rs" + str(random_seed),
    random_seed=random_seed,
)


###
# train and save
###
print("   training")
model.train(n_epochs=epochs, train_minimum=train_minimum, developer_mode=True, stop_after=stop_after)
model.save()
print("   model saved")

###
# predict for test set
###
original_name = model._model_name
# change the model name (because we did inference once for 10 epochs and once for 50)
model._model_name = original_name + "_test10e"
testset = adata[adata.obs["train_val_test"] == "test"].copy()
model.predict_new(testset)
print("   test set inferred")

model._model_name = original_name + "_test50e"
model.predict_new(testset, n_epochs=50)
print("   test set inferred (long)")