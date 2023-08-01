import mudata as md
import pandas as pd

from omicsdgd import DGD

# define seed in command line
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=0)
args = parser.parse_args()
random_seed = args.random_seed

###
# load data
###
# test different feature selection thresholds
import anndata as ad
import numpy as np

gex = ad.read_h5ad("data/raw/anndata.h5ad")
modality_switch = gex.X.shape[1]
atac = ad.read_h5ad("data/raw/PeakMatrix_anndata.h5ad")
ids_shared = list(
    set(gex.obs["sample"].index.values).intersection(
        set(atac.obs["sample"].index.values)
    )
)
ids_gex = np.where(gex.obs["sample"].index.isin(ids_shared))[0]
ids_atac = np.where(atac.obs["sample"].index.isin(ids_shared))[0]
gex = gex[ids_gex]
atac = atac[ids_atac]
threshold = 0.00
mudata = md.MuData({"rna": gex, "atac": atac})
mudata.obs["stage"] = mudata["atac"].obs["stage"]  # .values
mudata.obs["celltype"] = mudata["rna"].obs["celltype"]  # .values
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
dgd_name = (
    "mouse_gast_l20_h3-3w2_rs"
    + str(random_seed)
    + "_scale5"
    + "_featselect"
    + str(threshold).split(".")[1]
)
model = DGD(
    data=mudata,
    parameter_dictionary=hyperparameters,
    train_validation_split=train_val_split,
    meta_label="celltype",
    correction="stage",
    save_dir="./results/trained_models/mouse_gastrulation/",
    model_name=dgd_name,
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

test_gex = np.array(testset.X[:, :modality_switch].todense())
test_recon = model.decoder_forward(model.test_rep.z.shape[0])
test_recon_gex = test_recon[0].cpu().detach().numpy()
test_recon_atac = test_recon[1].cpu().detach().numpy()
np.save(
    "results/analysis/performance_evaluation/reconstruction/"
    + dgd_name
    + "_test_recon_gex.npy",
    test_recon_gex,
)
np.save(
    "results/analysis/performance_evaluation/reconstruction/"
    + dgd_name
    + "_test_recon_atac.npy",
    test_recon_atac,
)
print("   saved reconstructions")
