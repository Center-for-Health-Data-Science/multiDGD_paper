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
parser.add_argument("--unpaired", type=float, default=0.0)
args = parser.parse_args()
random_seed = args.random_seed
epochs = args.epochs
train_minimum = args.train_minimum
stop_after = args.stop_after
fraction_unpaired = args.unpaired
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

valset = adata[adata.obs["train_val_test"] == "validation"].copy()
valset.obs["modality"] = "paired"
testset = adata[adata.obs["train_val_test"] == "test"].copy()

###
# prep unpaired data if not already existing
###
# if the unpairing dataframe does not exist yet, create it
import os
if not os.path.exists('../../data/'+data_name+'_unpairing.csv'):
    fraction_unpaired_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    n_samples = len(train_val_split[0])

    df_out = pd.DataFrame({
        'sample_idx': train_val_split[0],
        'fraction_unpaired': [0]*n_samples,
        'modality': ['paired']*n_samples
    })
    for fraction_unpaired in fraction_unpaired_options:
        # select random samples to be unpaired
        unpairing = np.random.choice(np.arange(n_samples), size=int(len(train_val_split[0])*fraction_unpaired), replace=False)
        mod_1_indices = unpairing[::2]
        mod_2_indices = unpairing[1::2]
        # make the string list
        modality = ['paired']*n_samples
        modality = ['rna' if i in mod_1_indices else x for i,x in enumerate(modality)]
        modality = ['atac' if i in mod_2_indices else x for i,x in enumerate(modality)]
        # make temp df
        df_temp = pd.DataFrame({
            'sample_idx': train_val_split[0],
            'fraction_unpaired': [fraction_unpaired]*n_samples,
            'modality': modality
        })
        df_out = pd.concat([df_out, df_temp], axis=0)
    df_out.to_csv('../../data/'+data_name+'_unpairing.csv', index=False)

df_unpaired = pd.read_csv('../../data/'+data_name+'_unpairing.csv')

mod_1_indices = df_unpaired[
    (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "rna")
]["sample_idx"].values
mod_2_indices = df_unpaired[
    (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "atac")
]["sample_idx"].values
remaining_indices = df_unpaired[
    (df_unpaired["fraction_unpaired"] == fraction_unpaired) & (df_unpaired["modality"] == "paired")
]["sample_idx"].values

var_before = adata.var.copy()
print(var_before)
if fraction_unpaired < 1.0:
    adata_rna = adata[mod_1_indices, adata.var["feature_types"] == "GEX"].copy()
    adata_rna.obs["modality"] = "GEX"
    print("copied rna")
    adata_atac = adata[mod_2_indices, adata.var["feature_types"] == "ATAC"].copy()
    adata_atac.obs["modality"] = "ATAC"
    print("copied atac")
    adata_multi = adata[remaining_indices, :].copy()
    adata_multi.obs["modality"] = "paired"
    print("copied rest")
    adata = None
    print("freed some memory")
    #adata_unpaired = scvi.data.organize_multiome_anndatas(adata_multi, adata_rna, adata_atac)
    #adata_unpaired = adata_multi.concatenate(adata_rna).concatenate(adata_atac)
    adata_unpaired = ad.concat([adata_multi, adata_rna, adata_atac], join="outer")
    print("organized data")
else:
    adata_unpaired = adata[mod_1_indices,:].copy()
    adata_unpaired.obs['modality'] = 'GEX'
    adata_temp = adata[mod_2_indices,:].copy()
    adata_temp.obs['modality'] = 'ATAC'
    #adata_unpaired = adata_unpaired.concatenate(adata_temp)
    adata_unpaired = ad.concat([adata_unpaired, adata_temp], join="outer")
    print("organized data")
    adata, adata_temp = None, None

adata_rna, adata_atac, adata_multi = None, None, None
print("freed memory")
#adata = adata_unpaired.concatenate(valset)
adata = ad.concat([adata_unpaired, valset], join="inner")
print("finished data")
adata.var = var_before

# update the train-val split
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
    "log_wandb": ["viktoriaschuster", "omicsDGD_revision"]
}

model = DGD(
    data=adata,
    parameter_dictionary=hyperparameters,
    train_validation_split=train_val_split,
    modalities="feature_types",
    meta_label="cell_type",
    correction="Site",
    save_dir="../results/trained_models/" + data_name + "/",
    model_name=data_name + "_l20_h2-3_rs" + str(random_seed)+"_mosaic"+str(fraction_unpaired)+"percent",
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
model.predict_new(testset)
print("   test set inferred")

model._model_name = original_name + "_test50e"
model.predict_new(testset, n_epochs=50)
print("   test set inferred (long)")