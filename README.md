# The code used to produce results for multiDGD

This is a separate repository from our tool [multiDGD](https://github.com/Center-for-Health-Data-Science/multiDGD), collecting all our code we used to produce the results presented in our manuscript. Experiments are ordered according to their appearance in the manuscript and contain code for training and analysing the model and for making the figures.

## Download trained models and data 

All models and processed datasets are available via [Figshare](https://figshare.com/articles/dataset/multiDGD_-_processed_data_and_models/23796198).

Download data and models:
```python
import requests
import zipfile

figshare_url = 'https://api.figshare.com/v2/articles/23796198/files'
files = {
    'human_bonemarrow.h5ad.zip':'41740251',                # processed data 
    'dgd_human_bonemarrow.pt':'41735907',                  # model weights
    'dgd_human_bonemarrow_hyperparameters.json':'41735904' # hyperparameters
}

for file_name, file_id in files.items():
    file_url = f'{figshare_url}/{file_id}'
    file_response = requests.get(file_url).json()
    file_download_url = file_response['download_url']
    response = requests.get(file_download_url, stream=True)
    with open(file_name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    if file_name.endswith('.zip'):     
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall('.')            
```

Load trained model:
```python
import scanpy as sc
import multiDGD

# Load data
data = sc.read_h5ad('./human_bonemarrow.h5ad')
# Patch for broken data loader
data.obs['train_val_test'].to_csv('./_obs.csv')

# load model from the saved checkpoint
model = multiDGD.DGD.load(data=data, save_dir='./', model_name='dgd_human_bonemarrow')
```