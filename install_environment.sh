# first create a conda environment
conda env create -f environment.yml
conda activate multidgd-paper
# some packages (anndata, scvi-tools etc) were throwing conflicts, so we install them in the environment
pip install -r requirements.txt

# now we can install the local package for this paper
python -m pip install ./src