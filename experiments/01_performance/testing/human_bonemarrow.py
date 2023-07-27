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

#random_seed = 8790
#random_seed = 37
data_name = 'human_bonemarrow'
if random_seed == 0:
    dgd_name = '_l20_h2-3'
else:
    dgd_name = '_l20_h2-3_rs'+str(random_seed)

trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
model = DGD.load(data=trainset,
            save_dir=save_dir+data_name+'/',
            model_name=data_name+dgd_name)
print('loaded model')

model.predict_new(testset)
print('new samples learned')