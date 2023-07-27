# figure 3: new data integration
# comparing DGD to MultiVI+scArches on what datasets?

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os
from omicsdgd import DGD
import umap.umap_ as umap
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
figure_height = 16
n_cols = 3
n_rows = 4
grid_wspace = 1.2
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
legend_x_dist, legend_y_dist = 0.02, 0.0
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
data_names = ["human_bonemarrow", "mouse_gastrulation", "human_brain"]
dgd_names = ['human_bonemarrow_l20_h2-3', 'mouse_gast_l20_h2-2_rs0', 'human_brain_l20_h2-2_a2_long']
mvi_names = ["l20_e2_d2", "l20_e2_d2", "l20_e1_d1"]
#dgd_neighbors = [50, 50, 50]
#dgd_dists = [0.75, 0.75, 0.75]
#mvi_neighbors = [50, 50, 50]
#mvi_dists = [0.75, 0.75, 0.75]
# try first default
dgd_neighbors = [15, 15, 15]
dgd_dists = [0.5, 0.5, 0.5]
mvi_neighbors = [15, 15, 15]
mvi_dists = [0.1, 0.1, 0.1]
dgd_grid_letters = ['A', 'B', 'C']
mvi_grid_letters = ['D', 'E', 'F']
plotting_key = ['cell_type', 'celltype', 'rna_celltype']
legendscale = 0.2
mvi_batch_keys = ['Site', 'stage']

#####################
# first row: DGD
#####################
print('doing DGD')

for i, data_name in enumerate(data_names):
    model_name = dgd_names[i]
    cluster_class_neworder, class_palette = make_palette_from_meta(data_name)
    column_names = ['UMAP D1', 'UMAP D2']
    if not os.path.exists('results/analysis/performance_evaluation/dgd_'+data_name+'_umap.csv'):
        print('making umap for data '+data_name)
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


        model = DGD.load(data=trainset,
                save_dir=save_dir+data_name+'/',
                model_name=model_name)
        # get latent spaces in reduced dimensionality
        rep = model.representation.z.detach().numpy()
        #if data_name != 'human_brain':
        #    correction_rep = model.correction_rep.z.detach().numpy()
        cell_labels = trainset.obs[plotting_key[i]].values
        #batch_labels = trainset.obs['Site'].values
        test_rep = model.test_rep.z.detach().numpy()

        # make umap
        n_neighbors = dgd_neighbors[i]
        min_dist = dgd_dists[i]
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist)
        projected = reducer.fit_transform(rep)
        plot_data = pd.DataFrame(projected, columns=column_names)
        plot_data['cell type'] = cell_labels
        plot_data['cell type'] = plot_data['cell type'].astype("category")
        plot_data['cell type'] = plot_data['cell type'].cat.set_categories(cluster_class_neworder)
        #plot_data['batch'] = batch_labels
        #plot_data['batch'] = plot_data['batch'].astype("category")
        #plot_data['batch'] = plot_data['batch'].cat.set_categories(['site1','site2','site3','site4'])
        plot_data['data set'] = 'train'
        #projected_test = reducer.transform(test_rep)
        #plot_data_test = pd.DataFrame(projected_test, columns=column_names)
        #plot_data_test['data set'] = 'test'
        #correction_df = pd.DataFrame(correction_rep, columns=['D1', 'D2'])
        #correction_df['batch'] = batch_labels
        #correction_df['batch'] = correction_df['batch'].astype("category")
        #correction_df['batch'] = correction_df['batch'].cat.set_categories(['site1','site2','site3','site4'])
        #train_test_df = plot_data.copy()
        #train_test_df.drop(columns=['cell type','batch'], inplace=True)
        #train_test_df = pd.concat([train_test_df, plot_data_test], axis=0)
        #train_test_df['data set'] = train_test_df['data set'].astype("category")
        #train_test_df['data set'] = train_test_df['data set'].cat.set_categories(['train','test'])
        # transform GMM means and save
        #projected_gmm = reducer.transform(model.gmm.mean.detach().numpy())
        #projected_gmm = pd.DataFrame(projected_gmm, columns=column_names)
        #projected_gmm['type'] = 'mean'
        #gmm_samples = reducer.transform(model.gmm.sample(10000).detach().numpy())
        #gmm_samples = pd.DataFrame(gmm_samples, columns=column_names)
        #gmm_samples['type'] = 'sample'
        #projected_gmm = pd.concat([projected_gmm, gmm_samples], axis=0)
        # save files
        plot_data.to_csv('results/analysis/performance_evaluation/dgd_'+data_name+'_umap.csv', index=False)
        #correction_df.to_csv('results/analysis/performance_evaluation/bonemarrow_correction_umap.csv', index=False)
        #train_test_df.to_csv('results/analysis/performance_evaluation/bonemarrow_train_test_umap.csv', index=False)
        #projected_gmm.to_csv('results/analysis/performance_evaluation/bonemarrow_gmm_umap.csv', index=False)
    else:
        plot_data = pd.read_csv('results/analysis/performance_evaluation/dgd_'+data_name+'_umap.csv')
        plot_data['cell type'] = plot_data['cell type'].astype("category")
        plot_data['cell type'] = plot_data['cell type'].cat.set_categories(cluster_class_neworder)
        #plot_data['batch'] = plot_data['batch'].astype("category")
        #plot_data['batch'] = plot_data['batch'].cat.set_categories(['site1','site2','site3','site4'])
        #correction_df = pd.read_csv('results/analysis/performance_evaluation/bonemarrow_correction_umap.csv')
        #correction_df['batch'] = correction_df['batch'].astype("category")
        #correction_df['batch'] = correction_df['batch'].cat.set_categories(['site1','site2','site3','site4'])
        #train_test_df = pd.read_csv('results/analysis/performance_evaluation/bonemarrow_train_test_umap.csv')
        #train_test_df['data set'] = train_test_df['data set'].astype("category")
        #train_test_df['data set'] = train_test_df['data set'].cat.set_categories(['train','test'])
        #projected_gmm = pd.read_csv('results/analysis/performance_evaluation/bonemarrow_gmm_umap.csv')
    
    ax_list.append(plt.subplot(gs[0,i]))
    ax_list[-1].text(grid_letter_positions[0], 1.+grid_letter_positions[1],
                dgd_grid_letters[i], transform=ax_list[-1].transAxes + trans,
                fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
                fontweight=grid_letter_fontweight)
    sns.scatterplot(data=plot_data.sort_values(by='cell type'),
                    x=column_names[0], y=column_names[1],
                    hue='cell type', palette=class_palette,
                    ax=ax_list[-1], 
                    s=point_size, alpha=alpha, linewidth=point_linewidth)
    # legend
    ax_list[-1].legend(bbox_to_anchor=(1.+legend_x_dist, 1.+legend_y_dist),
                  loc='upper left', frameon=False,
                  handletextpad=handletextpad, markerscale=handlesize, fontsize=3)
    ax_list[-1].set_title('DGD ('+data_name.replace('_', ' ')+')')
    # remove axis ticks
    ax_list[-1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    # also remove axis tick values
    ax_list[-1].set_xticklabels([])
    ax_list[-1].set_yticklabels([])

###################
# second row: MVI
###################

print('doing MVI')

for i, data_name in enumerate(data_names):
    model_name = mvi_names[i]
    cluster_class_neworder, class_palette = make_palette_from_meta(data_name)
    column_names = ['UMAP D1', 'UMAP D2']
    if not os.path.exists('results/analysis/performance_evaluation/mvi_'+data_name+'_umap.csv'):
        print('making umap for data '+data_name)
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
        rep = model.get_latent_representation()
        model = None
        cell_labels = trainset.obs[plotting_key[i]].values

        # make umap
        n_neighbors = mvi_neighbors[i]
        min_dist = mvi_dists[i]
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist)
        projected = reducer.fit_transform(rep)
        plot_data = pd.DataFrame(projected, columns=column_names)
        plot_data['cell type'] = cell_labels
        plot_data['cell type'] = plot_data['cell type'].astype("category")
        plot_data['cell type'] = plot_data['cell type'].cat.set_categories(cluster_class_neworder)
        plot_data['data set'] = 'train'
        plot_data.to_csv('results/analysis/performance_evaluation/mvi_'+data_name+'_umap.csv', index=False)
    else:
        plot_data = pd.read_csv('results/analysis/performance_evaluation/mvi_'+data_name+'_umap.csv')
        plot_data['cell type'] = plot_data['cell type'].astype("category")
        plot_data['cell type'] = plot_data['cell type'].cat.set_categories(cluster_class_neworder)
    
    ax_list.append(plt.subplot(gs[1,i]))
    ax_list[-1].text(grid_letter_positions[0], 1.+grid_letter_positions[1],
                mvi_grid_letters[i], transform=ax_list[-1].transAxes + trans,
                fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
                fontweight=grid_letter_fontweight)
    sns.scatterplot(data=plot_data.sort_values(by='cell type'),
                    x=column_names[0], y=column_names[1],
                    hue='cell type', palette=class_palette,
                    ax=ax_list[-1], 
                    s=point_size, alpha=alpha, linewidth=point_linewidth)
    # legend
    ax_list[-1].legend(bbox_to_anchor=(1.+legend_x_dist, 1.+legend_y_dist),
                  loc='upper left', frameon=False,
                  handletextpad=handletextpad, markerscale=handlesize, fontsize=3).remove()
    ax_list[-1].set_title('MVI ('+data_name.replace('_', ' ')+')')
    # remove axis ticks
    ax_list[-1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    # also remove axis tick values
    ax_list[-1].set_xticklabels([])
    ax_list[-1].set_yticklabels([])

plot_data = pd.read_csv('results/analysis/performance_evaluation/cobolt_human_brain_rs0_umap.csv')
plot_data['cell type'] = plot_data['cell type'].astype("category")
plot_data['cell type'] = plot_data['cell type'].cat.set_categories(cluster_class_neworder)
ax_list.append(plt.subplot(gs[2,2]))
ax_list[-1].text(grid_letter_positions[0], 1.+grid_letter_positions[1],
            "G", transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)
sns.scatterplot(data=plot_data.sort_values(by='cell type'),
                x=column_names[0], y=column_names[1],
                hue='cell type', palette=class_palette,
                ax=ax_list[-1], 
                s=point_size, alpha=alpha, linewidth=point_linewidth)
ax_list[-1].legend().remove()
ax_list[-1].set_title('Cobolt ('+data_name.replace('_', ' ')+')')
# remove axis ticks
ax_list[-1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
# also remove axis tick values
ax_list[-1].set_xticklabels([])
ax_list[-1].set_yticklabels([])

plot_data = pd.read_csv('results/analysis/performance_evaluation/scmm_human_brain_rs0_umap.csv')
plot_data['cell type'] = plot_data['cell type'].astype("category")
plot_data['cell type'] = plot_data['cell type'].cat.set_categories(cluster_class_neworder)
ax_list.append(plt.subplot(gs[3,2]))
ax_list[-1].text(grid_letter_positions[0], 1.+grid_letter_positions[1],
            "H", transform=ax_list[-1].transAxes + trans,
            fontsize=grid_letter_fontsize, va='bottom', fontfamily=grid_letter_fontfamily,
            fontweight=grid_letter_fontweight)
sns.scatterplot(data=plot_data.sort_values(by='cell type'),
                x=column_names[0], y=column_names[1],
                hue='cell type', palette=class_palette,
                ax=ax_list[-1], 
                s=point_size, alpha=alpha, linewidth=point_linewidth)
ax_list[-1].legend().remove()
ax_list[-1].set_title('scMM ('+data_name.replace('_', ' ')+')')
# remove axis ticks
ax_list[-1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
# also remove axis tick values
ax_list[-1].set_xticklabels([])
ax_list[-1].set_yticklabels([])

figure_name = "results/analysis/plots/fig_embeddings_supp_2"
plt.savefig(figure_name+".png", dpi=300, bbox_inches="tight")
