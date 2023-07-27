'''
This script creates a dataframe of train indices to use for subsets of the data
(for reproducibility and fairness between methods)
'''

##############
# imports
##############

import pandas as pd
import numpy as np

##############
# define fractions
##############
fraction_options = [0.01, 0.1, 0.25, 0.5, 0.75]

# get train test split to know how many samples are in the train set
data_name = 'human_bonemarrow'
is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
train_indices = is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values
n_samples = len(train_indices)

for fraction in fraction_options:
    # select random samples to be unpaired
    chosen_ones = np.random.choice(np.arange(n_samples), size=int(len(train_indices)*fraction), replace=False)
    print(fraction, len(chosen_ones))
    # make the string list
    decision_list = [0]*n_samples
    decision_list = [1 if i in chosen_ones else x for i,x in enumerate(decision_list)]
    # make temp df
    df_temp = pd.DataFrame({
        'sample_idx': train_indices,
        'fraction': [fraction]*n_samples,
        'include': decision_list
    })
    if fraction == fraction_options[0]:
        df_out = df_temp
    else:
        df_out = pd.concat([df_out, df_temp], axis=0)
df_out.to_csv('data/'+data_name+'/data_subsets.csv', index=False)