import scvi
import pandas as pd
import anndata as ad
from omicsdgd import DGD
from omicsdgd.functions._data_manipulation import load_testdata_as_anndata

save_dir = 'results/trained_models/'
data_name = 'human_bonemarrow'

'''
Go through datasets and chosen models and compute reconstruction performances
'''
adata = ad.read_h5ad('data/'+data_name+'/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
#trainset_all = trainset.copy()
trainset = None
df_subset_ids = pd.read_csv('data/'+data_name+'/data_subsets.csv')

subset_samples = [
    #567,
    #5671,
    #14178,
    28357,
    #42535,
    #56714
]
subset_samples.reverse()
fraction_options = [
    #0.01,
    #0.1,
    #0.25,
    0.5,
    #0.75,
    #1.0
]
fraction_options.reverse()

for count, fraction in enumerate(fraction_options):
    print('###')
    print(fraction)
    print('###')
    subset = subset_samples[count]
    
    if fraction == 1.0:
        n_samples = len(trainset_all)
    else:
        train_indices = list(df_subset_ids[(df_subset_ids['fraction'] == fraction) & (df_subset_ids['include'] == 1)]['sample_idx'].values)
        trainset = adata[train_indices].copy()
        n_samples = len(train_indices)
    print('loaded data')

    for random_seed in [8790]:#[0,37,8790]:
        print('###')
        print(random_seed)
        print('###')
        model_name = 'human_bonemarrow_l20_h2-3_rs'+str(random_seed)+'_subset'+str(subset)
        if fraction == 1.0:
            if random_seed == 0:
                model_name = 'human_bonemarrow_l20_h2-3_test50e'
            else:
                model_name = 'human_bonemarrow_l20_h2-3_rs'+str(random_seed)
        
        if fraction == 1.0:
            model = DGD.load(data=trainset_all,
                    save_dir=save_dir+data_name+'/',
                    model_name=model_name)
        else:
            model = DGD.load(data=trainset,
                    save_dir=save_dir+data_name+'/',
                    model_name=model_name)
        model.init_test_set(testset)
        predictions = model.predict_from_representation(model.test_rep, model.correction_test_rep)
        loss = model.get_prediction_errors(predictions, model.test_set)
        print(loss.item())
        metrics_temp = pd.DataFrame({
            'loss': [loss.item()],
            'n_samples': [n_samples],
            'fraction': [fraction],
            'model': ['multiDGD'],
            'random_seed': [random_seed]
        })
        model = None
        #if (count == 0) & (random_seed == 0):
        #    metrics_df = metrics_temp
        #else:
        #    metrics_df = metrics_df.append(metrics_temp)
    
    if fraction == 1.0:
        trainset_all = None
    else:
        trainset = None

#metrics_df.to_csv('results/analysis/performance_evaluation/'+data_name+'_data_efficiency.csv')

print('done')