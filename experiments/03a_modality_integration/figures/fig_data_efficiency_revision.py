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

save_dir = '../results/trained_models/'

####################################
# plotting specs
####################################

# set up figure
figure_height = 6
n_cols = 1
n_rows = 1
cm = 1/2.54
fig = plt.figure(figsize=(6*cm,figure_height*cm))
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

data_name = 'human_bonemarrow'
#model_name = 'human_bonemarrow_l20_h2-3_rs0_unpaired0percent'
model_name = "human_bonemarrow_l20_h2-3_test50e"

ax_list.append(plt.subplot(gs[0]))

umap_df = pd.read_csv("../results/analysis/modality_integration/human_bonemarrow_l20_h2-3_rs0_unpaired0percent_latent_integration_umap_all.csv")
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

plt.savefig('../results/revision/plots/fig_data_efficiency_revision.png', dpi=720, bbox_inches='tight')