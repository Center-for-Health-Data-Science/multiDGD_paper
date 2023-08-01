# Experiments

Our experiments folder is structured according to the order in which they appear in our manuscript. Every folder contains subfolders for training and testing models (if applicable), analysis and figures.

### 01_performance

This folder refers to Sections `2.2 Improved performance on data reconstruction and clustering` and `2.3 Probabilistic modelling of batch effects`. It contains most training and analysis in this work.
Pretrained models are available on Figshare under [https://doi.org/10.6084/m9.figshare.23796198.v1](https://doi.org/10.6084/m9.figshare.23796198.v1).

If you would like to try to reproduce anything, please stick to the order of the folders and run `bash_run.sh` before `bash_analysis.sh`.

### 02_efficiency

Refers to `2.4 High performance on small data sets and many features`.

### 03a_modality_integration

Refers to `2.5.1 Predicting missing modalities`.

### 03b_batch_integration

Refers to `2.5.2 Integrating unseen batches without architectural surgery`.

### 04_gene2peak

Refers to `2.6 Gene-to-peak association with in silico perturbation`.