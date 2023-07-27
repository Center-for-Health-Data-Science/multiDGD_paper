from omicsdgd import DGD

from omicsdgd.functions._data_manipulation import load_testdata_as_anndata
from omicsdgd.functions import set_random_seed

# create argument parser for random seed
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=0)
args = parser.parse_args()
random_seed = args.random_seed
set_random_seed(random_seed)

save_dir = 'results/trained_models/'

data_name = 'human_brain'
if random_seed == 0:
    dgd_name = 'human_brain_l20_h2-2_a2_long'
else:
    dgd_name = 'human_brain_l20_h2-2_a2_rs'+str(random_seed)
trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
trainset.obs['celltype'] = trainset.obs['atac_celltype']
testset.obs['celltype'] = testset.obs['atac_celltype']

model = DGD.load(data=trainset,
            save_dir=save_dir+data_name+'/',
            model_name=dgd_name)

model.predict_new(testset)
print('new samples learned')