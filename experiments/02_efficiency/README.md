## 02_efficiency

This folder refers to Section `2.4 High performance on small data sets and many features`.

### Training

Script `020_bash_run_data_efficiency.sh` runs the training of multiDGD and MultiVI for different subset sizes of the human bone marrow data. Script `021_bash_run_feature_efficiency.sh` runs the training of multiDGD and MultiVI for different numbers of features of the mouse gastrulation data.

### Analysis

Script `022_bash_analysis_data_efficiency.sh` runs the analysis of the results of the training of multiDGD and MultiVI for different subset sizes of the human bone marrow data. The figure is part of the figure for section `03a`. Script `023_bash_analysis_feature_efficiency.sh` prepares the ground truth test counts for the mouse gastrulation set, and the results (from manuscript table 1) are computed in [this notebook](https://github.com/Center-for-Health-Data-Science/multiDGD_paper/blob/main/experiments/02_efficiency/analysis/mouse_reconstruction_feature_selection.ipynb).

## Execution

```
bash 020_bash_run_data_efficiency.sh
bash 021_bash_run_feature_efficiency.sh
bash 022_bash_analysis_data_efficiency.sh
bash 023_bash_analysis_feature_efficiency.sh
```
Then execute [this notebook](https://github.com/Center-for-Health-Data-Science/multiDGD_paper/blob/main/experiments/02_efficiency/analysis/mouse_reconstruction_feature_selection.ipynb).

## !Note!

***This is still a work in progress, and we hope you have patience with us. We are cleaning and refactoring our code from more than half a year of work for you, so that you can run the experiments from our manuscript.***