import mudata as md
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
parser.add_argument("--n_components", type=int)
args = parser.parse_args()
random_seed = args.random_seed
epochs = args.epochs
train_minimum = args.train_minimum
stop_after = args.stop_after
n_components = args.n_components
print("training multiDGD on mouse gastrulation data with random seed ", random_seed)
print("   and n_components: ", n_components)

###
# load data
###
mudata = md.read("../../data/mouse_gastrulation.h5mu", backed=False)
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
    "log_wandb": ["viktoriaschuster", "omicsDGD_revision"],
    "n_components": n_components,
}

model = DGD(
    data=mudata,
    parameter_dictionary=hyperparameters,
    train_validation_split=train_val_split,
    meta_label="celltype",
    correction="stage",
    save_dir="../results/trained_models/mouse_gastrulation/",
    model_name="mouse_gast_l20_h2-2_rs" + str(random_seed)+"_ncomp"+str(n_components),
    random_seed=random_seed,
)

###
# train and save
###
model.train(n_epochs=epochs, train_minimum=train_minimum, developer_mode=True, stop_after=stop_after)
model.save()
print("   model saved")

###
# predict for test set
###
testset = mudata[mudata.obs["train_val_test"] == "test"].copy()
mudata = None
###
# predict for test set
###
original_name = model._model_name
# change the model name (because we did inference once for 10 epochs and once for 50)
model._model_name = original_name + "_test10e"
model.predict_new(testset)
print("   test set inferred")

model._model_name = original_name + "_test50e"
model.predict_new(testset, n_epochs=50)
print("   test set inferred (long)")