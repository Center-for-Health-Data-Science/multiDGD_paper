import scvi
import pandas as pd
from omicsdgd import DGD
import numpy as np

from omicsdgd.functions._analysis import testset_reconstruction_evaluation
from omicsdgd.functions._data_manipulation import load_testdata_as_anndata

save_dir = 'results/trained_models/'
data_names = ['human_bonemarrow', 'mouse_gastrulation', 'human_brain']
model_names_all = [
    [
        ['l20_e2_d2','l20_e2_d2_rs37','l20_e2_d2_rs8790'],
        ['human_bonemarrow_l20_h2-3','human_bonemarrow_l20_h2-3_rs37','human_bonemarrow_l20_h2-3_rs8790']
    ], # all models for human bone marrow
    [
        ['l20_e2_d2','l20_e2_d2_rs37','l20_e2_d2_rs8790'],
        #['mouse_gast_l20_h2-2_c20_new2','mouse_gast_l20_h2-2_c20_new2_rs37','mouse_gast_l20_h2-2_c20_new2_rs8790']
        ['mouse_gast_l20_h2-2_rs0','mouse_gast_l20_h2-2_rs37','mouse_gast_l20_h2-2_rs8790']
    ],
    [
        ['l20_e1_d1','l20_e1_d1_rs37','l20_e1_d1_rs8790'],
        ['human_brain_l20_h2-2_a2_long','human_brain_l20_h2-2_a2_rs37','human_brain_l20_h2-2_a2_rs8790']
    ]
]
data_index = 1
data_name = data_names[data_index]
model_names = model_names_all[data_index]
random_seeds = [0,37,8790]

'''
Go through datasets and chosen models and compute reconstruction performances
'''

trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
print(testset.X.toarray()[1,:10])
print('loaded data')

# multiDGD first
for count, model_name in enumerate(model_names[1]):
    print(model_name)

    # compute for DGD
    model = DGD.load(data=trainset,
            save_dir=save_dir+data_name+'/',
            model_name=model_name)
    print('loaded model')
    if data_name == 'human_brain':
        metrics_temp = testset_reconstruction_evaluation(testset, model, modality_switch, library, thresholds=[0.2], batch_size=32)
    else:
        metrics_temp = testset_reconstruction_evaluation(testset, model, modality_switch, library, thresholds=[0.2])
    metrics_temp['model'] = 'multiDGD'
    metrics_temp['random_seed'] = random_seeds[count]
    model = None
    if count == 0:
        metrics_dgd = metrics_temp
    else:
        metrics_dgd = metrics_dgd.append(metrics_temp)
    #metrics_dgd.to_csv('results/analysis/performance_evaluation/reconstruction/'+data_name+'.csv')

# compute for multiVI
if data_name == 'mouse_gastrulation':
    trainset.var_names_make_unique()
    trainset.obs['modality'] = 'paired'
    #trainset.obs['_indices'] = np.arange(trainset.n_obs)
    scvi.model.MULTIVI.setup_anndata(trainset, batch_key='stage')
    testset.var_names_make_unique()
    testset.obs['modality'] = 'paired'
    #testset.obs['_indices'] = np.arange(testset.n_obs)
    scvi.model.MULTIVI.setup_anndata(testset, batch_key='stage')
elif data_name == 'human_bonemarrow':
    trainset.var_names_make_unique()
    trainset.obs['modality'] = 'paired'
    scvi.model.MULTIVI.setup_anndata(trainset, batch_key='Site')
    testset.var_names_make_unique()
    testset.obs['modality'] = 'paired'
    scvi.model.MULTIVI.setup_anndata(testset, batch_key='Site')
else:
    trainset.var_names_make_unique()
    trainset.obs['modality'] = 'paired'
    scvi.model.MULTIVI.setup_anndata(trainset)
    testset.var_names_make_unique()
    testset.obs['modality'] = 'paired'
    scvi.model.MULTIVI.setup_anndata(testset)

for count, model_name in enumerate(model_names[0]):
    print(model_name)
    #'''
    model = scvi.model.MULTIVI.load(
        save_dir+'multiVI/'+data_name+'/'+model_name,
        adata=trainset
    )
    if data_name == 'human_brain':
        metrics_temp = testset_reconstruction_evaluation(testset, model, modality_switch, library, batch_size=32)
    else:
        metrics_temp = testset_reconstruction_evaluation(testset, model, modality_switch, library)
    metrics_temp['model'] = 'multiVI'
    metrics_temp['random_seed'] = random_seeds[count]
    model = None
    if count == 0:
        metrics_mvi = metrics_temp
    else:
        metrics_mvi = metrics_mvi.append(metrics_temp)
    #'''

metrics_df = pd.concat([metrics_mvi, metrics_dgd])
metrics_df.to_csv('results/analysis/performance_evaluation/reconstruction/'+data_name+'_4.csv')

print('done')