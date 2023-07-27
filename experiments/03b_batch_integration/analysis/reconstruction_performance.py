import scvi
import pandas as pd
from omicsdgd import DGD
import numpy as np

from omicsdgd.functions._analysis import testset_reconstruction_evaluation
from omicsdgd.functions._data_manipulation import load_testdata_as_anndata

save_dir = 'results/trained_models/'
data_names = ['human_bonemarrow']
model_names_all = [
    [
        ['l20_e2_d2_leftout_site1_scarches','l20_e2_d2_leftout_site2_scarches','l20_e2_d2_leftout_site3_scarches','l20_e2_d2_leftout_site4_scarches'],
        ['human_bonemarrow_l20_h2-3_leftout_site1','human_bonemarrow_l20_h2-3_leftout_site2','human_bonemarrow_l20_h2-3_leftout_site3','human_bonemarrow_l20_h2-3_leftout_site4']
    ]
]
data_index = 0
data_name = data_names[data_index]
model_names = model_names_all[data_index]

'''
Go through datasets and chosen models and compute reconstruction performances
'''

is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
print('loaded data')

# multiDGD first
for count, model_name in enumerate(model_names[1]):
    print(model_name, model_name.split('_')[-1])
    batches = trainset.obs['Site'].unique()
    train_indices = [x for x in np.arange(len(trainset)) if trainset.obs['Site'].values[x] != model_name.split('_')[-1]]

    # compute for DGD
    model = DGD.load(data=trainset[train_indices],
            save_dir=save_dir+data_name+'/',
            model_name=model_name)
    print('loaded model')
    metrics_temp = testset_reconstruction_evaluation(testset, model, modality_switch, library, thresholds=[0.2])
    metrics_temp['model'] = 'multiDGD'
    metrics_temp['batch'] = model_name.split('_')[-1]
    model = None
    if count == 0:
        metrics_dgd = metrics_temp
    else:
        metrics_dgd = metrics_dgd.append(metrics_temp)
    #metrics_dgd.to_csv('results/analysis/performance_evaluation/reconstruction/'+data_name+'.csv')

# compute for multiVI
trainset.var_names_make_unique()
trainset.obs['modality'] = 'paired'
scvi.model.MULTIVI.setup_anndata(trainset, batch_key='Site')
testset.var_names_make_unique()
testset.obs['modality'] = 'paired'
scvi.model.MULTIVI.setup_anndata(testset, batch_key='Site')

for count, model_name in enumerate(model_names[0]):
    print(model_name, model_name.split('_')[-2])
    #'''
    model = scvi.model.MULTIVI.load(
        save_dir+'multiVI/'+data_name+'/'+model_name,
        adata=trainset
    )
    metrics_temp = testset_reconstruction_evaluation(testset, model, modality_switch, library)
    metrics_temp['model'] = 'multiVI'
    metrics_temp['batch'] = model_name.split('_')[-2]
    model = None
    if count == 0:
        metrics_mvi = metrics_temp
    else:
        metrics_mvi = metrics_mvi.append(metrics_temp)
    #'''

metrics_df = pd.concat([metrics_mvi, metrics_dgd])
metrics_df.to_csv('results/analysis/batch_integration/'+data_name+'_reconstruction_performance.csv')

print('done')