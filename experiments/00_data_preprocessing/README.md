## 00_data_preprocessing

The final data sets used for training models are available on Figshare under [https://doi.org/10.6084/m9.figshare.23796198.v1](https://doi.org/10.6084/m9.figshare.23796198.v1). You can just download these into a `./data/` subdirectory of the project, unzip them and go ahead with experiments. This is recommended for reproducibility. 

However, we provided the code for preparing these data sets. You can download the raw data with executing

```
00_downmoad_data.sh
```

and process them with 

```
01_preprocess_data.sh
```

from this directory.