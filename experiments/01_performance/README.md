## 01_performance

This folder refers to Sections `2.2 Improved performance on data reconstruction and clustering` and `2.3 Probabilistic modelling of batch effects`. It contains most training and analysis in this work.
Pretrained models are available on Figshare under [https://doi.org/10.6084/m9.figshare.23796198.v1](https://doi.org/10.6084/m9.figshare.23796198.v1).

### Training

Script `010_bash_run.sh` contains all commands to train multiDGD and MultiVI for all three data sets and three random seeds used. For training, it is highly recommended to split this into several parallel runs so it does not have to run for a week. Cobolt and scMM were trained separately on Google Colab. The notebooks are contained in `./training/`.

### Analysis

Script `011_bash_analysis.sh` collects the script executions for computing reconstruction errors, adjusted Rand indices and average silhouette widths.

### Figures

Figures can be generated based on the intermediate results via `012_bash_plot.sh`.

## !Note!

***This is still a work in progress, and we hope you have patience with us. We are cleaning and refactoring our code from more than half a year of work for you, so that you can run the experiments from our manuscript.***