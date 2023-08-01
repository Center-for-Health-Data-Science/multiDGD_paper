import mudata as md
import pandas as pd
import numpy as np

from omicsdgd import DGD

# define seed in command line
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=0)
args = parser.parse_args()
random_seed = args.random_seed
print("training multiDGD on mouse gastrulation data with random seed ", random_seed)

###
# load data
###
mudata = md.read("data/mouse_gastrulation.h5mu", backed=False)
train_val_split = [
    list(np.where(mudata.obs["train_val_test"] == "train")[0]),
    list(np.where(mudata.obs["train_val_test"] == "validation")[0]),
]

###
# initialize model
###
hyperparameters = {
    "latent_dimension": 20,
    "n_hidden": 2,
    "n_hidden_modality": 2,
    "log_wandb": ["viktoriaschuster", "omicsDGD"],
}

model = DGD(
    data=mudata,
    parameter_dictionary=hyperparameters,
    train_validation_split=train_val_split,
    meta_label="celltype",
    correction="stage",
    save_dir="./results/trained_models/mouse_gastrulation/",
    model_name="mouse_gast_l20_h2-2_rs" + str(random_seed),
    random_seed=random_seed,
)

###
# train and save
###
model.train(n_epochs=1000, train_minimum=100, developer_mode=True, stop_after=20)
model.save()
print("   model saved")

###
# predict for test set
###
testset = mudata[mudata.obs["train_val_test"] == "test"].copy()
mudata = None
model.predict_new(testset)
print("   test set inferred")
