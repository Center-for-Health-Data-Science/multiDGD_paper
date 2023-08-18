## 04_gene2peak

This subfolder contains notebooks that were used to generate results in Section `2.6 Gene-to-peak association with in silico perturbation`.

If you are intending to reproduce the results, please run the notebooks in the following order:
1. `./analysis/HiChIP_processing.ipynb` (prepares the EIS peak signal data used in distal association predictions)
2. `./analysis/distal_perturbations.ipynb` (performs the in silico knowdowns of ID2, CLEC16A and CD69 and returns predicted changes in harmoized chromatin regions with the EIS data)