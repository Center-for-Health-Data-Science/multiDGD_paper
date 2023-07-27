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

#print(plt.rcParams.keys())
#exit()

save_dir = 'results/trained_models/'
data_name = 'human_bonemarrow'
model_name = 'human_bonemarrow_l20_h2-3_test10e'

# set up figure
figure_height = 18
n_cols = 8
n_rows = 12
cm = 1/2.54
fig = plt.figure(figsize=(18*cm,figure_height*cm))
gs = gridspec.GridSpec(n_rows,n_cols)
gs.update(wspace = 6., hspace = 40.)
ax_list = []
#palette_2colrs = ['palegoldenrod', 'cornflowerblue']
palette_2colrs = ["#DAA327", "#015799"]
palette_models = ["#DAA327", "palegoldenrod", '#BDE1CD', "#015799"]
palette_3colrs = ["#DAA327", "#BDE1CD", "#015799"]
batch_palette = ['#EEE7A8', 'cornflowerblue', 'darkmagenta', 'darkslategray']
extra_palette = ['gray','darkslategray','#EEE7A8','#BDE1CD']
plt.rcParams.update({'font.size': 6, 'axes.linewidth': 0.3, 'xtick.major.size': 1.5, 'xtick.major.width': 0.3, 'ytick.major.size': 1.5, 'ytick.major.width': 0.3})
handletextpad = 0.1
legend_x_dist, legend_y_dist = -0.0, 0.0
grid_letter_positions = [-0.1, 0.05]
grid_letter_fontsize = 8
grid_letter_fontfamily = 'sans-serif'
grid_letter_fontweight = 'bold'
heatmap_fontsize = 4
point_size = 0.2
linewidth = 0.5
alpha = 1
point_linewidth = 0.0
handlesize = 0.3

# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)

####################################
####################################
# row 1: performance metrics (comparison to multiVI)
####################################
####################################
# load needed analysis data
clustering_df = pd.read_csv('results/analysis/performance_evaluation/clustering_and_batch_effects_multiome.csv')
clustering_df['ARI'] = clustering_df['ARI'].round(2)
clustering_df['ASW'] = clustering_df['ASW'].round(2)
clustering_cobolt = pd.read_csv('results/analysis/performance_evaluation/cobolt_human_brain_aris.csv')
clustering_cobolt['data'] = 'brain (H)'
clustering_cobolt['ARI'] = clustering_cobolt['ARI'].round(2)
multimodel_clustering = pd.concat([clustering_df, clustering_cobolt], axis=0)
clustering_scmm = pd.read_csv('results/analysis/performance_evaluation/scmm_human_brain_aris.csv')
clustering_scmm['ARI'] = clustering_scmm['ARI'].round(2)
multimodel_clustering = pd.concat([multimodel_clustering, clustering_scmm], axis=0)
multimodel_clustering['data'] = [x.split(' (')[0] for x in multimodel_clustering['data'].values]
multimodel_clustering['data'] = ['marrow' if x == 'bone marrow' else x for x in multimodel_clustering['data'].values]

# change multiVI to MultiVI and cobolt to Cobolt
multimodel_clustering['model'] = ['MultiVI' if x == 'multiVI' else x for x in multimodel_clustering['model'].values]
multimodel_clustering['model'] = ['Cobolt' if x == 'cobolt' else x for x in multimodel_clustering['model'].values]
multimodel_clustering['model'] = multimodel_clustering['model'].astype("category")
multimodel_clustering['model'] = multimodel_clustering['model'].cat.set_categories(["MultiVI", "Cobolt", "scMM", "multiDGD"])

reconstruction_temp = pd.read_csv('results/analysis/performance_evaluation/reconstruction/human_bonemarrow_3.csv')
reconstruction_df = reconstruction_temp
reconstruction_temp['data'] = 'marrow'
reconstruction_temp = pd.read_csv('results/analysis/performance_evaluation/reconstruction/mouse_gastrulation_3.csv', sep=';')
#reconstruction_temp = pd.read_csv('results/analysis/performance_evaluation/reconstruction/mouse_gastrulation_3.csv')
reconstruction_temp['data'] = 'gastrulation'
reconstruction_df = pd.concat([reconstruction_df, reconstruction_temp], axis=0)
#reconstruction_df = reconstruction_df.append(reconstruction_temp)
reconstruction_temp = pd.read_csv('results/analysis/performance_evaluation/reconstruction/human_brain_2.csv')
reconstruction_temp['data'] = 'brain'
reconstruction_df = pd.concat([reconstruction_df, reconstruction_temp], axis=0)
reconstruction_temp = pd.read_csv('results/analysis/performance_evaluation/reconstruction/scMM_brain_recon_performance.csv')
reconstruction_temp['data'] = 'brain'
# sort the data columns by the order of reconstruction_df
reconstruction_temp = reconstruction_temp[reconstruction_df.columns]
reconstruction_df = pd.concat([reconstruction_df, reconstruction_temp], axis=0)
#reconstruction_df = reconstruction_df.append(reconstruction_temp)
reconstruction_df = reconstruction_df.drop(columns=['random_seed', 'binary_threshold'])
reconstruction_df.reset_index(drop=True, inplace=True)
#performance_df = pd.concat([clustering_df, reconstruction_df], axis=1)
#print(performance_df)
#performance_df = performance_df.round(4)
#performance_df['data'] = [x.split(' (')[0] for x in performance_df['data'].values]
#performance_df['data'] = ['marrow' if x == 'bone marrow' else x for x in performance_df['data'].values]
#performance_df['model'] = ['MultiVI' if x == 'multiVI' else x for x in performance_df['model'].values]
#reconstruction_df['data'] = [x.split(' (')[0] for x in reconstruction_df['data'].values]
#reconstruction_df['data'] = ['marrow' if x == 'bone marrow' else x for x in reconstruction_df['data'].values]
reconstruction_df['model'] = ['MultiVI' if x == 'multiVI' else x for x in reconstruction_df['model'].values]
# set the order of the models to 'MultiVI', 'scMM', 'multiDGD'
reconstruction_df['model'] = reconstruction_df['model'].astype("category")
reconstruction_df['model'] = reconstruction_df['model'].cat.set_categories(["MultiVI", "scMM", "multiDGD"])

###############
# reconstruction performance (RMSE, Accuracy, ...) of DGD (also make binary atac trained version) and MultiVI
###############
pointplot_scale = 0.5
pointplot_errwidth = 0.7
pointplot_capsize = 0.2
ax_list.append(plt.subplot(gs[0:3,0:2]))
# label the first row as A
ax_list[-1].text(grid_letter_positions[0]*2, 1.+2*grid_letter_positions[1],
            'A', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)

#from brokenaxes import brokenaxes
#bax = brokenaxes(ylims=((0.6, 3.2), (4.5, 7.0)), hspace=.05, subplot_spec=ax_list[-1].get_subplotspec())
#sns.barplot(data=performance_df, x='data', y='RMSE (rna)', hue='model', ax=ax_list[-1], palette=palette_2colrs)
#sns.boxplot(data=performance_df, x='data', y='RMSE (rna)', hue='model', ax=ax_list[-1], palette=palette_2colrs, linewidth=linewidth)
sns.pointplot(data=reconstruction_df, x='data', y='RMSE (rna)', hue='model',
            ax=ax_list[-1], 
            palette=palette_3colrs, errorbar='se', dodge=0.3,
            markers='.', linestyles='', scale=pointplot_scale, errwidth=pointplot_errwidth, capsize=pointplot_capsize)

#ax_list[-1].set_ylim((0.6,3.2))
# rotate x labels by 45 degrees
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_xlabel('Dataset')
ax_list[-1].set_ylabel('Root-Mean-Square Error')
ax_list[-1].set_title('Reconstruction (RNA) \u2193')
ax_list[-1].legend().remove()
#    bbox_to_anchor=(1.05, 1.),
#    loc='upper left',
#    frameon=False).set_visible(False)

ax_list.append(plt.subplot(gs[0:3,2:4]))
ax_list[-1].text(grid_letter_positions[0]*2, 1.+2*grid_letter_positions[1],
            'B', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)
#sns.barplot(data=performance_df, x='data', y='balanced accuracy', hue='model', ax=ax_list[-1], palette=palette_2colrs)
#sns.boxplot(data=performance_df, x='data', y='balanced accuracy', hue='model', ax=ax_list[-1], palette=palette_2colrs, linewidth=linewidth)
sns.pointplot(data=reconstruction_df, x='data', y='balanced accuracy', hue='model',
              ax=ax_list[-1], palette=palette_3colrs, errorbar='se', dodge=0.3,
              markers='.', linestyles='', scale=pointplot_scale, errwidth=pointplot_errwidth, capsize=pointplot_capsize)
#ax_list[-1].set_ylim((0.66,0.82))
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_xlabel('Dataset')
ax_list[-1].set_ylabel('Balanced accuracy')
ax_list[-1].set_title('Reconstruction (ATAC) \u2191')
ax_list[-1].legend(bbox_to_anchor=(1.05, 1.), loc='upper left', frameon=False).set_visible(False)

###############
# clustering performances fo DGD, MultiVI (and ?) on all 4 datasets
###############
ax_list.append(plt.subplot(gs[0:3,4:6]))
ax_list[-1].text(grid_letter_positions[0]*2, 1.+2*grid_letter_positions[1],
            'C', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)
#sns.barplot(data=performance_df, x='data', y='ARI', hue='model', ax=ax_list[-1], palette=palette_2colrs)
#sns.boxplot(data=performance_df, x='data', y='ARI', hue='model', ax=ax_list[-1], palette=palette_2colrs, linewidth=linewidth)
sns.pointplot(data=multimodel_clustering, x='data', y='ARI', hue='model',
              ax=ax_list[-1], palette=palette_models, errorbar='se', dodge=0.3,
              markers='.', linestyles='', scale=pointplot_scale, errwidth=pointplot_errwidth, capsize=pointplot_capsize)
ax_list[-1].set_ylim((0.37,0.74))
ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_ylabel('Adjusted Rand Index')
ax_list[-1].set_xlabel('Dataset')
ax_list[-1].set_title('Clustering \u2191')
#ax_list[-1].legend(bbox_to_anchor=(1.05, 1.), loc='upper left', frameon=False).set_visible(False)
ax_list[-1].legend(bbox_to_anchor=(1.5+legend_x_dist, -0.3+legend_y_dist),
                    #bbox_to_anchor=(1.1+legend_x_dist, -0.3+legend_y_dist), 
                    loc='upper left', frameon=False, title='model',
                    alignment='left',
                    ncol=2,
                    columnspacing=0.3,
                    handletextpad=handletextpad)#.set_visible(False)

###############
# Batch effect removal metrics (ASW, ?)
###############
multimodel_clustering['ASW'] = [1 - x for x in multimodel_clustering['ASW'].values]
ax_list.append(plt.subplot(gs[0:3,6:]))
ax_list[-1].text(grid_letter_positions[0]*2, 1.+2*grid_letter_positions[1],
            'D', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)
#sns.boxplot(data=performance_df[performance_df['data'] != 'brain'], x='data', y='ASW', hue='model', ax=ax_list[-1], palette=palette_2colrs, linewidth=linewidth)
sns.pointplot(data=multimodel_clustering[multimodel_clustering['data'] != 'brain'], x='data', y='ASW', hue='model',
              ax=ax_list[-1], palette=palette_2colrs, errorbar='se', dodge=0.3,
              markers='.', linestyles='', scale=pointplot_scale, errwidth=pointplot_errwidth, capsize=pointplot_capsize)
ax_list[-1].set_ylim((0.98,1.11))
#ax_list[-1].set_xticklabels(ax_list[-1].get_xticklabels(), rotation=45)
ax_list[-1].set_ylabel('1 - ASW')
ax_list[-1].set_xlabel('Dataset')
ax_list[-1].set_title('Batch effect removal \u2191')
ax_list[-1].legend(bbox_to_anchor=(-0.2+legend_x_dist, -0.3+legend_y_dist), 
                   loc='upper left', frameon=False, title='model',
                   ncol=2,
                   columnspacing=0.3,
                   handletextpad=handletextpad).set_visible(False)
print('finished row 1: performance metrics')

####################################
####################################
# block lower left: example latent space visualization
####################################
####################################
# load model for this and next block
cluster_class_neworder, class_palette = make_palette_from_meta(data_name)
column_names = ['UMAP D1', 'UMAP D2']
if not os.path.exists('results/analysis/performance_evaluation/bonemarrow_umap.csv'):
    is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
    trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
    model = DGD.load(data=trainset,
            save_dir=save_dir+data_name+'/',
            model_name=model_name)
    # get latent spaces in reduced dimensionality
    rep = model.representation.z.detach().numpy()
    correction_rep = model.correction_rep.z.detach().numpy()
    cell_labels = trainset.obs['cell_type'].values
    batch_labels = trainset.obs['Site'].values
    test_rep = model.test_rep.z.detach().numpy()

    # make umap
    n_neighbors = 50
    min_dist = 0.75
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist)
    projected = reducer.fit_transform(rep)
    plot_data = pd.DataFrame(projected, columns=column_names)
    plot_data['cell type'] = cell_labels
    plot_data['cell type'] = plot_data['cell type'].astype("category")
    plot_data['cell type'] = plot_data['cell type'].cat.set_categories(cluster_class_neworder)
    plot_data['batch'] = batch_labels
    plot_data['batch'] = plot_data['batch'].astype("category")
    plot_data['batch'] = plot_data['batch'].cat.set_categories(['site1','site2','site3','site4'])
    plot_data['data set'] = 'train'
    projected_test = reducer.transform(test_rep)
    plot_data_test = pd.DataFrame(projected_test, columns=column_names)
    plot_data_test['data set'] = 'test'
    correction_df = pd.DataFrame(correction_rep, columns=['D1', 'D2'])
    correction_df['batch'] = batch_labels
    correction_df['batch'] = correction_df['batch'].astype("category")
    correction_df['batch'] = correction_df['batch'].cat.set_categories(['site1','site2','site3','site4'])
    train_test_df = plot_data.copy()
    train_test_df.drop(columns=['cell type','batch'], inplace=True)
    train_test_df = pd.concat([train_test_df, plot_data_test], axis=0)
    train_test_df['data set'] = train_test_df['data set'].astype("category")
    train_test_df['data set'] = train_test_df['data set'].cat.set_categories(['train','test'])
    # transform GMM means and save
    projected_gmm = reducer.transform(model.gmm.mean.detach().numpy())
    projected_gmm = pd.DataFrame(projected_gmm, columns=column_names)
    projected_gmm['type'] = 'mean'
    gmm_samples = reducer.transform(model.gmm.sample(10000).detach().numpy())
    gmm_samples = pd.DataFrame(gmm_samples, columns=column_names)
    gmm_samples['type'] = 'sample'
    projected_gmm = pd.concat([projected_gmm, gmm_samples], axis=0)
    # save files
    plot_data.to_csv('results/analysis/performance_evaluation/bonemarrow_umap.csv', index=False)
    correction_df.to_csv('results/analysis/performance_evaluation/bonemarrow_correction_umap.csv', index=False)
    train_test_df.to_csv('results/analysis/performance_evaluation/bonemarrow_train_test_umap.csv', index=False)
    projected_gmm.to_csv('results/analysis/performance_evaluation/bonemarrow_gmm_umap.csv', index=False)
else:
    plot_data = pd.read_csv('results/analysis/performance_evaluation/bonemarrow_umap.csv')
    plot_data['cell type'] = plot_data['cell type'].astype("category")
    plot_data['cell type'] = plot_data['cell type'].cat.set_categories(cluster_class_neworder)
    plot_data['batch'] = plot_data['batch'].astype("category")
    plot_data['batch'] = plot_data['batch'].cat.set_categories(['site1','site2','site3','site4'])
    correction_df = pd.read_csv('results/analysis/performance_evaluation/bonemarrow_correction_umap.csv')
    correction_df['batch'] = correction_df['batch'].astype("category")
    correction_df['batch'] = correction_df['batch'].cat.set_categories(['site1','site2','site3','site4'])
    train_test_df = pd.read_csv('results/analysis/performance_evaluation/bonemarrow_train_test_umap.csv')
    train_test_df['data set'] = train_test_df['data set'].astype("category")
    train_test_df['data set'] = train_test_df['data set'].cat.set_categories(['train','test'])
    projected_gmm = pd.read_csv('results/analysis/performance_evaluation/bonemarrow_gmm_umap.csv')

ax_list.append(plt.subplot(gs[4:8,0:3]))
# label the first row as B
ax_list[-1].text(grid_letter_positions[0]+0.01, 1.+grid_letter_positions[1]-0.01,
            'E', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)
sns.scatterplot(
    data=plot_data.sort_values(by='cell type'), x=column_names[0], y=column_names[1],
    hue='cell type', palette=class_palette,
    ax=ax_list[-1],s=point_size, alpha=alpha, linewidth=point_linewidth, rasterized=True)
# remove axis ticks
ax_list[-1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
# also remove axis tick values
ax_list[-1].set_xticklabels([])
ax_list[-1].set_yticklabels([])
ax_list[-1].set_title('latent representation (train set)')
ax_list[-1].legend(bbox_to_anchor=(1.0+legend_x_dist, 1.15+legend_y_dist),
                   loc='upper left',
                   frameon=False,
                   handletextpad=handletextpad*2,
                   markerscale=handlesize,
                   ncol=1,title='bone marrow cell type',
                   labelspacing=0.2)
for i in range(len(projected_gmm[projected_gmm['type'] == 'mean'])):
    ax_list[-1].text(projected_gmm[projected_gmm['type'] == 'mean'].iloc[i][column_names[0]],
                     projected_gmm[projected_gmm['type'] == 'mean'].iloc[i][column_names[1]],
                     str(i), fontsize=6, color='black', ha='center', va='center', fontweight='bold',
                     path_effects=[PathEffects.withStroke(linewidth=0.5, foreground='w')])

ax_list.append(plt.subplot(gs[9:,0:2]))
ax_list[-1].text(grid_letter_positions[0]*2, 1.+2*grid_letter_positions[1],
            'G', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)
sns.scatterplot(
    data=correction_df.sort_values(by='batch'), x='D1', y='D2',
    hue='batch', palette=batch_palette, alpha=alpha, linewidth=point_linewidth,
    ax=ax_list[-1],s=point_size)
ax_list[-1].set_title('batch representation')
# move the legend to one row in the bottom
ax_list[-1].legend(bbox_to_anchor=(-0.5, -0.3),
                    loc='upper left',
                    alignment='left',
                    frameon=False,
                    handletextpad=handletextpad,
                    markerscale=handlesize,
                    columnspacing=0.1,
                    title='batch',
                    ncol=4)#.set_visible(False)

ax_list[-1].set_xlabel(r'$Z^{cov}$ D1')
ax_list[-1].set_ylabel(r'$Z^{cov}$ D2')

"""
ax_list.append(plt.subplot(gs[5:,0:2]))
ax_list[-1].text(grid_letter_positions[0]*2, 1.+2*grid_letter_positions[1],
            'G', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)

umap_df = pd.read_csv("results/analysis/modality_integration/human_bonemarrow_l20_h2-3_rs0_unpaired0percent_latent_integration_umap_all.csv")
umap_df['data_set'] = umap_df['data_set'].astype('category')
umap_df['data_set'] = umap_df['data_set'].cat.set_categories(['train', 'test (paired)', 'test (atac)', 'test (rna)'])
sns.scatterplot(data=umap_df.sort_values(by='data_set'), x='UMAP1', y='UMAP2', hue='data_set', palette=extra_palette,
                s=point_size, ax=ax_list[-1], alpha=0.5, linewidth=point_linewidth,)
# move the legend to one row in the bottom
ax_list[-1].legend(bbox_to_anchor=(-0.4, -0.25),
                    loc='upper left',
                    frameon=False,
                    handletextpad=handletextpad,
                    markerscale=handlesize,
                    # minimize distances between handles and labels and between items
                    #labelspacing=0.1,
                    columnspacing=0.1,
                    ncol=4)
ax_list[-1].set_title("latent representation\nand integration of single modalities")

ax_list.append(plt.subplot(gs[5:,2:4]))
ax_list[-1].text(grid_letter_positions[0]*2, 1.+2*grid_letter_positions[1],
            'H', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)
sns.scatterplot(
    data=plot_data.sample(frac=1),
    x=column_names[0], y=column_names[1],
    hue='batch', palette=batch_palette, alpha=alpha, linewidth=point_linewidth,
    ax=ax_list[-1],s=point_size)
ax_list[-1].set_title('latent representation\n(train set)')
# move the legend to one row in the bottom
ax_list[-1].legend(bbox_to_anchor=(-0.12, -0.25),
                    loc='upper left',
                    frameon=False,
                    handletextpad=handletextpad,
                    markerscale=handlesize,
                    # minimize distances between handles and labels and between items
                    #labelspacing=0.2,
                    columnspacing=0.1,
                    ncol=4)
print('finished block 2: latent space visualization')
"""

####################################
####################################
# block lower right: cluster map
####################################
####################################
ax_list.append(plt.subplot(gs[4:,5:]))
# label the first row as C
ax_list[-1].text(grid_letter_positions[0]-0.35, 1.+grid_letter_positions[1]-0.03,
            'F', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)
# compute clustering of trained latent space
if not os.path.exists('results/analysis/performance_evaluation/bonemarrow_cluster_map.csv'):
    is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
    trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
    model = DGD.load(data=trainset,
            save_dir=save_dir+data_name+'/',
            model_name=model_name)
    df_relative_clustering = gmm_make_confusion_matrix(model)
    df_relative_clustering.to_csv('results/analysis/performance_evaluation/bonemarrow_cluster_map.csv')
else:
    df_relative_clustering = pd.read_csv('results/analysis/performance_evaluation/bonemarrow_cluster_map.csv', index_col=0)
if not os.path.exists('results/analysis/performance_evaluation/bonemarrow_cluster_map_nonorm.csv'):
    is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
    trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
    model = DGD.load(data=trainset,
            save_dir=save_dir+data_name+'/',
            model_name=model_name)
    df_clustering = gmm_make_confusion_matrix(model, norm=False)
    df_clustering.to_csv('results/analysis/performance_evaluation/bonemarrow_cluster_map_nonorm.csv')
else:
    df_clustering = pd.read_csv('results/analysis/performance_evaluation/bonemarrow_cluster_map_nonorm.csv', index_col=0)

#print(df_relative_clustering)
#print(df_clustering)
#""" version 1: heatmap with annotation
# prepare annotations (percentage of cluster represented by each cell type)
annotations = df_relative_clustering.to_numpy(dtype=np.float64).copy()
# reduce annotations for readability
annotations[annotations < 1] = None
df_relative_clustering = df_relative_clustering.fillna(0)
# choose color palette
cmap = sns.color_palette("GnBu", as_cmap=True)
# remove color bar
sns.heatmap(df_relative_clustering, annot=annotations,
    cmap=cmap, annot_kws={'size': heatmap_fontsize},
    cbar_kws={'shrink': 0.5, 'location': 'bottom'}, 
    xticklabels=True, yticklabels=True,
    mask=np.isnan(annotations),
    ax=ax_list[-1], alpha=0.8)
cbar = ax_list[-1].collections[0].colorbar
cbar.remove()
#ylabels = [df_clustering.index[x]+' ('+str(int(df_clustering.sum(axis=1)[x]))+')' for x in range(df_clustering.shape[0])]
#ax_list[-1].set(yticklabels=ylabels)
# rotate x labels and enforce every label to be shown
ax_list[-1].tick_params(axis='x', rotation=90, labelsize=heatmap_fontsize)
ax_list[-1].set_ylabel('Cell type')
ax_list[-1].set_xlabel('GMM component ID')
ax_list[-1].set_title('percentage of cell type in GMM cluster')
#"""
"""
# version 2: circles
ax_list[-1].grid(color = 'lightgray', linestyle = '-', linewidth = 0.1)
from matplotlib.collections import PatchCollection
xlabels = df_clustering.columns
ylabels = df_clustering.index
x,y = np.meshgrid(np.arange(df_clustering.shape[1]), np.arange(df_clustering.shape[0]))
s = df_clustering.to_numpy(dtype=np.float64).copy()
c = df_relative_clustering.to_numpy(dtype=np.float64).copy()
#R = s/s.max()/2
R = np.log10(s+1)/np.log10(s.max())/2
circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
col = PatchCollection(circles, array=c.flatten(), cmap='GnBu', alpha=0.9) # for color legend
#size = PatchCollection(circles, array=s.flatten()) # for size legend
ax_list[-1].add_collection(col)
ax_list[-1].set(xticks=np.arange(df_clustering.shape[1]),
                yticks=np.arange(df_clustering.shape[0]),
                xticklabels=xlabels,
                yticklabels=ylabels)
ax_list[-1].set_xticks(np.arange(df_clustering.shape[1]+1)-.5, minor=True)
ax_list[-1].set_yticks(np.arange(df_clustering.shape[0]+1)-.5, minor=True)
ax_list[-1].set_ylabel('Cell type')
ax_list[-1].set_xlabel('GMM component ID')
fig.colorbar(col, ax=ax_list[-1], label='tbd', shrink=0.95)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
size_legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='R max',
           markerfacecolor='gray', markersize=20),
    Line2D([0], [0], marker='o', color='w', label='R min',
           markerfacecolor='gray', markersize=2)]
#size_legend_elements = [
#    plt.Circle((0,0), radius=R.max(), color='gray', label='R max'),
#    plt.Circle((0,0), radius=R.min(), color='gray', label='R min')]
ax_list[-1].legend(handles=size_legend_elements,
                   bbox_to_anchor=(1.22+legend_x_dist, 1.+legend_y_dist), 
                   loc='upper left', frameon=False,
                   handletextpad=handletextpad)
"""
print('finished block 3: cluster map')

####################################
####################################
# last subplot: data efficiency on human bonemarrow
####################################
####################################
ax_list.append(plt.subplot(gs[9:,2:4]))
ax_list[-1].text(grid_letter_positions[0]*2, 1.+2*grid_letter_positions[1],
            'H', transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)
"""
ax_list[-1].set_title('data efficiency')

# should be loss ratio to 100% on the y axis
# colored by model
# x axis: percentage of data set
efficiency_df_mvi = pd.read_csv('results/analysis/performance_evaluation/human_bonemarrow_data_efficiency_mvi.csv')
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

sns.pointplot(data=efficiency_df, x='fraction', y='loss ratio',
             hue='model', palette=palette_2colrs, ax=ax_list[-1],
             capsize=0.2,linestyles=['--', '--'],
             scale=0.5,errwidth=0.8
             #linewidth=0.5,markersize=1.,errwidth=0.5
             )
ax_list[-1].legend(bbox_to_anchor=(1.02+legend_x_dist, 1.+legend_y_dist),
                   loc='upper left', frameon=False,
                   handletextpad=handletextpad*4)#.set_visible(False)
#ax_list[-1].text(30, 1.01, 'placeholder', fontdict={'color': 'red'})
ax_list[-1].set_xlabel('Percentage of training set')
print('finished block 4: data efficiency')
"""
print("getting mouse correction rep")
if not os.path.exists('results/analysis/performance_evaluation/gastrulation_correction.csv'):
    trainset, testset, modality_switch, library = load_testdata_as_anndata("mouse_gastrulation")
    model = DGD.load(data=trainset,
            save_dir=save_dir+'mouse_gastrulation/',
            model_name="mouse_gast_l20_h2-2_rs0")
    # get latent spaces in reduced dimensionality
    correction_rep = model.correction_rep.z.detach().numpy()
    batch_labels = trainset.obs['stage'].values
    correction_df = pd.DataFrame(correction_rep, columns=['D1', 'D2'])
    correction_df['batch'] = batch_labels
    correction_df['batch'] = correction_df['batch'].astype("category")
    correction_df.to_csv('results/analysis/performance_evaluation/gastrulation_correction.csv', index=False)
else:
    correction_df = pd.read_csv('results/analysis/performance_evaluation/gastrulation_correction.csv')
    correction_df['batch'] = correction_df['batch'].astype("category")
#batch_palette = ['palegoldenrod', 'cornflowerblue', 'coral', 'darkmagenta', 'darkslategray']
sns.scatterplot(
    data=correction_df.sort_values(by='batch'), x='D1', y='D2',
    hue='batch', palette='magma_r',#palette=batch_palette,
    ax=ax_list[-1],s=point_size, alpha=alpha, linewidth=point_linewidth,)
ax_list[-1].set_title('gastrulation stage rep.')

ax_list[-1].legend(bbox_to_anchor=(1.02+legend_x_dist, 1.+legend_y_dist),
                   loc='upper left',
                   frameon=False,
                   markerscale=handlesize,
                   handletextpad=handletextpad,
                   title='stage',
                   ncol=1)
ax_list[-1].set_xlabel(r'$Z^{cov}$ D1')
ax_list[-1].set_ylabel(r'$Z^{cov}$ D2')

####################################
####################################
# save figure
####################################
####################################

plt.savefig('results/analysis/plots/performance_evaluation/fig2_v5.png', dpi=300, bbox_inches='tight')