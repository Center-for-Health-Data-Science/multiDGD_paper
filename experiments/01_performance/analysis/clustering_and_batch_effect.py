#import torch
#checkpoint = torch.load('results/trained_models/mouse_gastrulation/mouse_gast_l20_h2-2.pt',map_location=torch.device('cpu'))
#print(checkpoint.keys())
#print(checkpoint['representation.z'].shape)

'''
output should be a dataframe with clustering performances 
of MultiVI and DGD (and another model specific for clustering)
for the 4 different datasets
'''

import pandas as pd

'''
data_names = ['human_pbmc', 'mouse_gastrulation']
model_names = [
    ['l20_e2_d2'],
    ['l20_h2-3', 'l20_h2-2_c20']
]

for data_index in range(len(data_names)):

    data_name = data_names[data_index]
    multivi_name = model_names[0][data_index]
    dgd_name = model_names[1][data_index]
'''

'''...I actually already got all ARIs...'''

clustering_df = pd.DataFrame({
    'data': [
        'bone marrow (H)', 'bone marrow (H)', 'bone marrow (H)', 'bone marrow (H)', 'bone marrow (H)', 'bone marrow (H)',
        'gastrulation (M)', 'gastrulation (M)', 'gastrulation (M)', 'gastrulation (M)', 'gastrulation (M)', 'gastrulation (M)',
        'brain (H)', 'brain (H)', 'brain (H)', 'brain (H)', 'brain (H)', 'brain (H)'
        ],
    'model': [
        'multiVI', 'multiVI', 'multiVI', 'multiDGD', 'multiDGD', 'multiDGD',
        'multiVI', 'multiVI', 'multiVI', 'multiDGD', 'multiDGD', 'multiDGD',
        'multiVI', 'multiVI', 'multiVI', 'multiDGD', 'multiDGD', 'multiDGD',
        ],
    'random seed': [
        0, 37, 8790, 0, 37, 8790,
        0, 37, 8790, 0, 37, 8790,
        0, 37, 8790, 0, 37, 8790,
        ],
    'ARI': [
        0.5398677100524277, 0.5594169010079396, 0.5421540523475306, 0.645189037173994, 0.5699, 0.6415,
        #0.6239633435576255, 0.5994804730720469, 0.5781245801509648, 0.7135, None, None,
        0.5112, 0.5152, 0.4987, 0.6178, 0.5063, 0.501,
        0.5819, 0.551457875566191, 0.5369323347711512, 0.6021, 0.6735, 0.5727,
        ],
    'ASW': [
        -0.012500444, -0.014410537, -0.017072814, -0.05824040621519089, -0.0592, -0.05756,
        #-0.033694357, -0.042453762, -0.04392023, -0.07201, None, None,
        -0.033694357, -0.042453762, -0.04392023, -0.07769, -0.09291, -0.05858,
        0,0,0,0,0,0
        ]
})

clustering_df.to_csv('results/analysis/performance_evaluation/clustering_and_batch_effects_multiome.csv', index=None)