# figure in manuscript
python ./figures/fig2_v2.py

# supplementaries
# first make sure there is a directory
if [ ! -d "../results/analysis/plots/supplementaries" ]; then
    mkdir ../results/analysis/plots/supplementaries
fi
python ./figures/supplementaries/fig_heatmaps.py
python ./figures/supplementaries/fig_heatmaps_marrow.py
python ./figures/supplementaries/fig_heatmaps_mvi_not_marrow.py
python ./figures/supplementaries/fig_embeddings_new.py
python ./figures/supplementaries/fig_embeddings_covariate.py
python ./figures/supplementaries/fig_embeddings_tcells.py