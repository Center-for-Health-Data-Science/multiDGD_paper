import os
import numpy as np
import umap.umap_ as umap
import scanpy as sc
import pandas as pd
import anndata as ad
from sklearn import preprocessing
from sklearn.metrics import adjusted_rand_score
import scipy
from omicsdgd.functions._analysis import make_palette_from_meta

# import the numpy representations derived from cobolt
save_dir = "../results/other_models/scMM/"
data_dir = "../../data/"
random_seeds = [0, 37, 8790]

########################
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

figure_height = 4
n_cols = 3
n_rows = 1
grid_wspace = 0.5
grid_hspace = 0.5
cm = 1 / 2.54
fig = plt.figure(figsize=(18 * cm, figure_height * cm))
gs = gridspec.GridSpec(n_rows, n_cols)
gs.update(wspace=grid_wspace, hspace=grid_hspace)
ax_list = []
plt.rcParams.update(
    {
        "font.size": 6,
        "axes.linewidth": 0.3,
        "xtick.major.size": 1.5,
        "xtick.major.width": 0.3,
        "ytick.major.size": 1.5,
        "ytick.major.width": 0.3,
    }
)
handletextpad = 0.1
handlesize = 0.3
########################

# get umaps
data_name = "human_brain"
column_names = ["UMAP D1", "UMAP D2"]
trainset = None
cluster_class_neworder, class_palette = make_palette_from_meta(data_name)
aris_out = []
for i, seed in enumerate(random_seeds):
    print("seed: ", seed)
    if not os.path.exists(
        save_dir+"scmm_"
        + data_name
        + "_rs"
        + str(random_seeds[i])
        + "_umap.csv"
    ):
        if trainset is None:
            import mudata as md
            data = md.read(data_dir+"human_brain.h5mu", backed=False)
            modality_switch = data["rna"].X.shape[1]
            adata = ad.AnnData(scipy.sparse.hstack((data["rna"].X, data["atac"].X)))
            adata.obs = data.obs
            adata.var = pd.DataFrame(
                index=data["rna"].var_names.tolist() + data["atac"].var_names.tolist(),
                data={
                    "name": data["rna"].var["name"].values.tolist()
                    + data["atac"].var["name"].values.tolist(),
                    "feature_types": ["rna"] * modality_switch
                    + ["atac"] * (adata.shape[1] - modality_switch),
                },
            )
            data = None
            data = adata
            train_indices = list(np.where(data.obs["train_val_test"] == "train")[0])
            test_indices = list(np.where(data.obs["train_val_test"] == "test")[0])
            if not isinstance(
                data.X, scipy.sparse.csc_matrix
            ):
                data.X = data.X.tocsr()
            trainset = data.copy()[train_indices]
            #trainset.obs["cell_type"] = trainset.obs["atac_celltype"].values
            cell_labels = trainset.obs["celltype"].values

        # load rep
        rep = pd.read_csv(
            save_dir + "scmm_lat_train_mean_rs" + str(random_seeds[i]) + ".csv",
            index_col=0,
        ).values
        print("   rep shape: ", rep.shape)

        # make umap
        print("   making umap")
        n_neighbors = 15
        min_dist = 0.1
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist)
        projected = reducer.fit_transform(rep)
        plot_data = pd.DataFrame(projected, columns=column_names)
        plot_data["cell type"] = cell_labels
        plot_data["cell type"] = plot_data["cell type"].astype("category")
        plot_data["cell type"] = plot_data["cell type"].cat.set_categories(
            cluster_class_neworder
        )
        plot_data["data set"] = "train"
        # compute clustering
        print("   clustering")
        trainset.obsm["latent"] = rep
        sc.pp.neighbors(trainset, use_rep="latent", n_neighbors=15)
        sc.tl.leiden(trainset, key_added="clusters", resolution=1)
        n_clusters = len(np.unique(trainset.obs["clusters"].values))
        print("   number of leiden clusters: ", n_clusters)
        le = preprocessing.LabelEncoder()
        le.fit(trainset.obs["celltype"].values)
        true_labels = le.transform(trainset.obs["celltype"].values)
        cluster_labels = trainset.obs["clusters"].values.astype(int)
        radj = adjusted_rand_score(true_labels, np.asarray(cluster_labels))
        print("   Adjusted RAND index: ", radj)
        plot_data["ARI"] = radj
        aris_out.append(radj)
        plot_data.to_csv(
            "../results/analysis/performance_evaluation/scmm_"
            + data_name
            + "_rs"
            + str(random_seeds[i])
            + "_umap.csv",
            index=False,
        )
    else:
        plot_data = pd.read_csv(
            "../results/analysis/performance_evaluation/scmm_"
            + data_name
            + "_rs"
            + str(random_seeds[i])
            + "_umap.csv"
        )

    ax_list.append(plt.subplot(gs[i]))
    ax_list[-1].set_title("seed " + str(seed))
    ax_list[-1].set_xlabel("UMAP D1")
    ax_list[-1].set_ylabel("UMAP D2")
    sns.scatterplot(
        data=plot_data,
        x="UMAP D1",
        y="UMAP D2",
        hue="cell type",
        palette=class_palette,
        ax=ax_list[-1],
        s=1,
        linewidth=0,
    )
    if i == len(random_seeds) - 1:
        ax_list[-1].legend(
            bbox_to_anchor=(1.0, 1.15),
            loc="upper left",
            frameon=False,
            handletextpad=handletextpad * 2,
            markerscale=handlesize,
            ncol=1,
            title="brain cell type",
            labelspacing=0.2,
        )
    else:
        ax_list[-1].legend().remove()
if len(aris_out) > 0:
    # save aris as data frame with columns 'data', 'model', 'random seed', 'ARI'
    aris_df = pd.DataFrame(columns=["data", "model", "random seed", "ARI"])
    aris_df["data"] = ["brain (H)"] * len(random_seeds)
    aris_df["model"] = ["scMM"] * len(random_seeds)
    aris_df["random seed"] = random_seeds
    aris_df["ARI"] = aris_out
    aris_df.to_csv(
        "../results/analysis/performance_evaluation/scmm_" + data_name + "_aris.csv",
        index=False,
    )

# make sure directory exists
if not os.path.exists("../results/analysis/plots/performance_evaluation/"):
    os.makedirs("../results/analysis/plots/performance_evaluation/")
plt.savefig(
    "../results/analysis/plots/performance_evaluation/fig_supp_scmm_latent.png",
    dpi=300,
    bbox_inches="tight",
)
