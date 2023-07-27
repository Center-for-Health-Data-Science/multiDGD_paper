'''
Explore what visually effective things to show at the end of Fig 2

I am thinking of comparing PCAs and umaps with same parameters for best models of DGD and multiVI
to show more complex latent space

I still think it would be good to compute structure preservation
'''

import os
import pandas as pd
from omicsdgd import DGD
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as PathEffects

from omicsdgd.functions._data_manipulation import load_testdata_as_anndata
from omicsdgd.functions._analysis import make_palette_from_meta, gmm_make_confusion_matrix

save_dir = 'results/trained_models/'
data_name = 'human_bonemarrow'
model_name = 'human_bonemarrow_l20_h2-3_rs0_unpaired0percent'

# set up figure
figure_height = 6
n_cols = 2
n_rows = 1
cm = 1/2.54
fig = plt.figure(figsize=(18*cm,figure_height*cm))
gs = gridspec.GridSpec(n_rows,n_cols)
gs.update(wspace = 0.6, hspace = 0.6)
ax_list = []
#palette_2colrs = ['palegoldenrod', 'cornflowerblue']
#extra_palette = ['gray','darkslategray','#EEE7A8','#BDE1CD']
extra_palette = ['gray','darkslategray','#EEE7A8','cornflowerblue']
#palette_2colrs = ['#DAA327', 'cornflowerblue']
palette_2colrs = ["#DAA327", "#015799"]
batch_palette = ['palegoldenrod', 'cornflowerblue', 'darkmagenta', 'darkslategray']
plt.rcParams.update({'font.size': 6, 'axes.linewidth': 0.3, 
    'xtick.major.size': 1.5, 'xtick.major.width': 0.3, 
    'ytick.major.size': 1.5, 'ytick.major.width': 0.3,
    'lines.linewidth': 1.0})
handletextpad = 0.1
legend_x_dist, legend_y_dist = -0.02, 0.0
grid_letter_positions = [-0.1, 0.2]
grid_letter_fontsize = 8
grid_letter_fontfamily = 'sans-serif'
grid_letter_fontweight = 'bold'
point_size = 1
handletextpad = 0.1
point_linewidth = 0.0
handlesize = 0.3

# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)

####################################
####################################
# data efficiency on human bonemarrow
####################################
####################################
ax_list.append(plt.subplot(gs[0]))
ax_list[-1].text(-0.1, 1.1,
            'A', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)
# should be loss ratio to 100% on the y axis
# colored by model
# x axis: percentage of data set
efficiency_df_mvi = pd.read_csv('results/analysis/performance_evaluation/human_bonemarrow_data_efficiency_mvi.csv')
# change multiVI to MultiVI in model column
efficiency_df_mvi['model'] = ['MultiVI' if x == 'multiVI' else x for x in efficiency_df_mvi['model'].values]
efficiency_df_mvi['fraction'] = efficiency_df_mvi['fraction'] * 100
loss_ratios = []
for count, loss in enumerate(efficiency_df_mvi['loss'].values):
    rs = efficiency_df_mvi['random_seed'].values[count]
    base_loss = efficiency_df_mvi[(efficiency_df_mvi['random_seed'] == rs) & (efficiency_df_mvi['fraction'] == 100)]['loss'].item()
    loss_ratios.append(loss / base_loss)
efficiency_df_mvi['loss ratio'] = loss_ratios
efficiency_df = pd.read_csv('results/analysis/performance_evaluation/human_bonemarrow_data_efficiency.csv', sep=';')
efficiency_df['fraction'] = efficiency_df['fraction'] * 100
loss_ratios = []
for count, loss in enumerate(efficiency_df['loss'].values):
    rs = efficiency_df['random_seed'].values[count]
    base_loss = efficiency_df[(efficiency_df['random_seed'] == rs) & (efficiency_df['fraction'] == 100)]['loss'].item()
    loss_ratios.append(loss / base_loss)
efficiency_df['loss ratio'] = loss_ratios
efficiency_df = pd.concat([efficiency_df_mvi, efficiency_df], axis=0)
# make fraction column integers
efficiency_df['fraction'] = efficiency_df['fraction'].astype(int)
efficiency_df['n_samples_k'] = [str(round(x/1000, 1))+'k' for x in efficiency_df['n_samples'].values]

sns.pointplot(data=efficiency_df, x='fraction', y='loss ratio',
             hue='model', palette=palette_2colrs, ax=ax_list[-1],
             capsize=0.2,linestyles=['--', '--'],
             scale=0.5,errwidth=0.8,errorbar='se'
             )
def percent_to_n(x):
    print(x)
    print(type(x))
    return int(x * 0.01 * 56714)
def n_to_percent(x):
    return int(x / 56714 * 100)
#secax = ax_list[-1].secondary_xaxis('top', functions=(percent_to_n, n_to_percent))
#secax.set_xlabel('Number of training samples')
ax2 = ax_list[-1].twiny()
sns.pointplot(data=efficiency_df.sort_values(by='n_samples'), x='n_samples_k', y='loss ratio',
             hue='model', palette=palette_2colrs, ax=ax2,
             capsize=0.2,linestyles=['--', '--'],
             scale=0.5,errwidth=0.8,errorbar='se'
             #linewidth=0.5,markersize=1.,errwidth=0.5
             )
ax2.legend_.remove()
ax_list[-1].legend(bbox_to_anchor=(1.0+legend_x_dist, 1.+legend_y_dist),
                   loc='upper left', frameon=False,title='model',
                   markerscale=handlesize*3,
                   handletextpad=handletextpad*2)#.set_visible(False)
#ax_list[-1].text(30, 1.01, 'placeholder', fontdict={'color': 'red'})
ax_list[-1].set_xlabel('Percentage of training set')
ax2.set_xlabel('Number of training samples')
#ax_list[-1].set_title("data efficiency\n")

####################################
####################################
# single modality integration
####################################

ax_list.append(plt.subplot(gs[1]))
ax_list[-1].text(-0.1, 1.1,
            'B', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)

umap_df = pd.read_csv("results/analysis/modality_integration/human_bonemarrow_l20_h2-3_rs0_unpaired0percent_latent_integration_umap_all.csv")
umap_df['data_set'] = umap_df['data_set'].astype('category')
umap_df['data_set'] = umap_df['data_set'].cat.set_categories(['train', 'test (paired)', 'test (atac)', 'test (rna)'])
sns.scatterplot(data=umap_df,#.sort_values(by='data_set'), 
                x='UMAP1', y='UMAP2', hue='data_set', palette=extra_palette,
                s=point_size, ax=ax_list[-1], alpha=0.5, linewidth=point_linewidth,)
# move the legend to one row in the bottom
ax_list[-1].legend(bbox_to_anchor=(1.0+legend_x_dist, 1.+legend_y_dist),
                    loc='upper left',
                    frameon=False,
                    handletextpad=handletextpad*2,
                    title='data set',
                    markerscale=handlesize)
                    # minimize distances between handles and labels and between items
                    #labelspacing=0.1)
ax_list[-1].set_title("Integration of single modalities")

####################################
####################################
# save figure
####################################
####################################

plt.savefig('results/analysis/plots/performance_evaluation/fig_data_efficiency_v2.png', dpi=720, bbox_inches='tight')