## 01_performance

This folder refers to Sections `2.2 Improved performance on data reconstruction and clustering` and `2.3 Probabilistic modelling of batch effects`. It contains most training and analysis in this work.
Pretrained models are available on Figshare under [https://doi.org/10.6084/m9.figshare.23796198.v1](https://doi.org/10.6084/m9.figshare.23796198.v1).

### Training

Script `bash_run.sh` contains all commands to train multiDGD and MultiVI for all three data sets and three random seeds used. For training, it is highly recommended to split this into several parallel runs so it does not have to run for a week. Cobolt and scMM were trained separately on Google Colab. The notebooks are contained in `./training/`.