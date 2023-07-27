# figure 3: new data integration
# comparing DGD to MultiVI+scArches on what datasets?

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os
import scanpy as sc
import matplotlib.transforms as mtransforms
from omicsdgd.functions._data_manipulation import load_testdata_as_anndata, load_data_from_name
from omicsdgd.functions._analysis import make_palette_from_meta

import anndata as ad
import scipy
import scvi

#####################
# define model names, directory and batches
#####################

####################################
# flexible parameters
####################################
figure_height = 8
n_cols = 3
n_rows = 1
grid_wspace = 0.7
grid_hspace = 0.5

####################################
# fixed figure design
####################################
# set up figure and grid
cm = 1/2.54
fig = plt.figure(figsize=(18*cm,figure_height*cm))
gs = gridspec.GridSpec(n_rows,n_cols)
gs.update(wspace = grid_wspace, hspace = grid_hspace)
ax_list = []
# fonts
# general text
text_size = 8
heatmap_fontsize = 3
plt.rcParams.update({'font.size': 4, 'axes.linewidth': 0.3, 'xtick.major.size': 1.5, 'xtick.major.width': 0.3, 'ytick.major.size': 1.5, 'ytick.major.width': 0.3})
# grid letters
grid_letter_positions = [-0.1, 0.05]
grid_letter_fontsize = 8
grid_letter_fontfamily = 'sans-serif'
grid_letter_fontweight = 'bold'
point_size = 1
# colors
palette_2colrs = ['palegoldenrod', 'cornflowerblue']
batch_palette = ['palegoldenrod', 'cornflowerblue', 'darkmagenta', 'darkslategray']
palette_3colrs = ["lightgray", "cornflowerblue", "darkmagenta", "darkolivegreen", "firebrick", "midnightblue"]
palette_continuous_1 = "GnBu"
palette_continuous_2 = 'magma_r'
# legend set up
legend_x_dist, legend_y_dist = 0.0, 0.0
handletextpad = 0.1
# scatter plot
point_size = 0.3
alpha = 1
point_linewidth = 0.0
handlesize = 0.2
# line plot
linewidth = 0.5

# set trans for labeling physical distance to the left and up:
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)

#####################
# load data
#####################
# first load the data for labels
save_dir = 'results/trained_models/'
data_names = ["mouse_gastrulation", "human_brain"]
mvi_names = ["l20_e2_d2", "l20_e1_d1"]
grid_letters = ['C', 'D']

plotting_key = ['celltype', 'rna_celltype']
mvi_batch_keys = ['stage']
leiden_resolutions = [2,1]
legendscale = 0.2

###################
# second row: MVI
###################

from sklearn import preprocessing
from omicsdgd.functions._analysis import confusion_matrix, get_connectivity_from_threshold, compute_distances, rank_distances, get_node_degrees, get_secondary_degrees, traverse_through_graph, order_matrix_by_max_per_class

def leiden_cluster_means(rep, clustering):
    # calculate means for subsets of data belonging to each cluster
    # rep: representation of data
    # clustering: cluster labels
    n_clusters = np.unique(clustering).shape[0]
    means = np.zeros((n_clusters, rep.shape[1]))
    for i in range(n_clusters):
        means[i,:] = np.mean(rep[clustering==i,:], axis=0)
    return means

def order_mvi_components(means):
    distance_mtrx = compute_distances(means)
    threshold = round(np.percentile(distance_mtrx.flatten(), 30), 2)

    connectivity_mtrx = get_connectivity_from_threshold(distance_mtrx, threshold)
    rank_mtrx = rank_distances(distance_mtrx)

    node_degrees = get_node_degrees(connectivity_mtrx)
    secondary_node_degrees = get_secondary_degrees(connectivity_mtrx)

    new_node_order = traverse_through_graph(connectivity_mtrx, rank_mtrx, node_degrees, secondary_node_degrees)
    return new_node_order

def mvi_confusion_matrix(mod, rep, labels, true_labels, cluster_labels, norm=True):
    classes = np.unique(labels)
    n_clusters = np.unique(cluster_labels).shape[0]
    print('n_clusters: ', n_clusters)
    cm1 = confusion_matrix(true_labels, cluster_labels)

    class_counts = [np.where(true_labels == i)[0].shape[0] for i in range(len(classes))]
    cm2 = cm1.astype(np.float64)
    for i in range(len(class_counts)):
        #percent_sum = 0
        for j in range(n_clusters):
            if norm:
                cm2[i,j] = (cm2[i,j]*100 / class_counts[i])
            else:
                cm2[i,j] = cm2[i,j]
            #percent_sum += (cm2[i,j]*100 / class_counts[i])
    cm2 = cm2.round()
    print('cm2 calculated')

    # get an order of components based on connectivity graph
    leiden_means = leiden_cluster_means(rep, cluster_labels)
    component_order = order_mvi_components(leiden_means)

    # take the non-empty entries
    cm2 = cm2[:len(classes),:n_clusters]

    cm3, classes_reordered = order_matrix_by_max_per_class(cm2, classes, component_order)
    out = pd.DataFrame(data=cm3, index=classes_reordered, columns=component_order)
    #out = pd.DataFrame(data=cm2, index=classes, columns=np.arange(n_clusters))
    return out

print('doing MVI')

for i, data_name in enumerate(data_names):
    print(data_name)
    model_name = mvi_names[i]
    if not os.path.exists('results/analysis/performance_evaluation/mvi_'+data_name+'_cluster_map.csv'):
        print('making clustering for data '+data_name)
        is_train_df = pd.read_csv('data/'+data_name+'/train_val_test_split.csv')
        if data_name != 'human_brain':
            trainset, testset, modality_switch, library = load_testdata_as_anndata(data_name)
        else:
            data = load_data_from_name(data_name)
            #print(data)
            modality_switch = data['rna'].X.shape[1]
            adata = ad.AnnData(scipy.sparse.hstack((data['rna'].X,data['atac'].X)))
            adata.obs = data['rna'].obs
            adata.var = pd.DataFrame(index=data['rna'].var_names.tolist()+data['atac'].var_names.tolist(),
                                    data={'name': data['rna'].var['name'].values.tolist()+data['atac'].var['name'].values.tolist(),
                                        'feature_types': ['rna']*modality_switch+['atac']*(adata.shape[1]-modality_switch)})
            data = None
            data = adata
            train_indices = is_train_df[is_train_df['is_train'] == 'train']['num_idx'].values
            test_indices = is_train_df[is_train_df['is_train'] == 'iid_holdout']['num_idx'].values
            if not isinstance(data.X, scipy.sparse.csc_matrix):# type(data.X) is not scipy.sparse._csc.csc_matrix:
                data.X = data.X.tocsr()
            trainset = data.copy()[train_indices]
            print(trainset)
        
        trainset.var_names_make_unique()
        trainset.obs['modality'] = 'paired'
        if data_name == 'human_bonemarrow':
            trainset.X = trainset.layers['counts']
        if data_name != 'human_brain':
            scvi.model.MULTIVI.setup_anndata(trainset, batch_key=mvi_batch_keys[i])
        else:
            scvi.model.MULTIVI.setup_anndata(trainset)
        model = scvi.model.MULTIVI.load(
                save_dir+'multiVI/'+data_name+'/'+model_name,
                adata=trainset
            )
        # get latent spaces in reduced dimensionality
        trainset.obsm['latent'] = model.get_latent_representation()

        sc.pp.neighbors(trainset, use_rep='latent', n_neighbors=15)
        sc.tl.leiden(trainset, key_added='clusters', resolution=leiden_resolutions[i])
        # get the number of clusters from leiden
        n_clusters = len(np.unique(trainset.obs['clusters'].values))
        print('number of leiden clusters: ', n_clusters)

        le = preprocessing.LabelEncoder()
        le.fit(trainset.obs[plotting_key[i]].values)
        true_labels = le.transform(trainset.obs[plotting_key[i]].values)
        print('true labels: ', true_labels)
        cluster_labels = trainset.obs['clusters'].values.astype(int)
        print('cluster labels: ', cluster_labels)

        # get confusion matrix
        df_relative_clustering = mvi_confusion_matrix(model, trainset.obsm['latent'], trainset.obs[plotting_key[i]].values, true_labels, cluster_labels)
        df_relative_clustering.to_csv('results/analysis/performance_evaluation/mvi_'+data_name+'_cluster_map.csv')

        df_clustering = mvi_confusion_matrix(model, trainset.obsm['latent'], trainset.obs[plotting_key[i]].values, true_labels, cluster_labels, norm=False)
        df_clustering.to_csv('results/analysis/performance_evaluation/mvi_'+data_name+'_cluster_map_nonorm.csv')
    else:
        df_relative_clustering = pd.read_csv('results/analysis/performance_evaluation/mvi_'+data_name+'_cluster_map.csv', index_col=0)
        df_clustering = pd.read_csv('results/analysis/performance_evaluation/mvi_'+data_name+'_cluster_map_nonorm.csv', index_col=0)
    
    if i == 0:
        ax_list.append(plt.subplot(gs[:2]))
    else:
        ax_list.append(plt.subplot(gs[2]))
    # label the first row as C
    extra_dist = 0.3
    if i > 0:
        extra_dist = 0.5
    ax_list[-1].text(grid_letter_positions[0]-extra_dist, 1.+grid_letter_positions[1],
                grid_letters[i], transform=ax_list[-1].transAxes + trans,
                fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
                fontweight=grid_letter_fontweight)
    annotations = df_relative_clustering.to_numpy(dtype=np.float64).copy()
    # reduce annotations for readability
    annotations[annotations < 1] = None
    df_relative_clustering = df_relative_clustering.fillna(0)
    # choose color palette
    cmap = sns.color_palette("GnBu", as_cmap=True)
    sns.heatmap(df_relative_clustering, annot=annotations,
    #sns.heatmap(df_clustering, annot=annotations,
        cmap=cmap, annot_kws={'size': heatmap_fontsize},
        mask=np.isnan(annotations),
        ax=ax_list[-1], alpha=0.8)
    ylabels = [df_clustering.index[x]+' ('+str(int(df_clustering.sum(axis=1)[x]))+')' for x in range(df_clustering.shape[0])]
    ax_list[-1].set(yticklabels=ylabels)
    ax_list[-1].tick_params(axis='x', rotation=90)
    ax_list[-1].tick_params(axis='y', rotation=0)
    ax_list[-1].set_ylabel('Cell type')
    ax_list[-1].set_xlabel('Leiden cluster ID')
    ax_list[-1].set_title('percentage of cell type represented by Leiden cluster ('+data_name+')')

plt.savefig('results/analysis/plots/performance_evaluation/fig_supp_heatmaps_mvi.png', dpi=300, bbox_inches='tight')