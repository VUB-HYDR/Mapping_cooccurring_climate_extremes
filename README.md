# Mapping of co-occurring climate extremes
This code further builds onto previous scripts from [Muheki et al., 2024](https://zenodo.org/records/10970524). Here, we map the global changes in the frequency, spatial distribution and dependence of co-occurring extreme events under different climate change scenarios. 

## Python Scripts
This repository includes two python scripts used in this anaylsis, namely:
### 1. [global_funcs.py](https://github.com/VUB-HYDR/Mapping_cooccurring_climate_extremes/blob/9307e114a7fd3c7d40915c570f0e6a3698ea877c/global_funcs.py)
This python script enntails all the functions used in this analysis. Users should first run this script, before running the main.py.

### 2. [global_main.py](https://github.com/VUB-HYDR/Mapping_cooccurring_climate_extremes/blob/5f5f7c851c614f35b492e51d907973d595fb165d/global_main.py)
This python script entails all the methods used to analyse the dataset using the functions within funcs.py. Users shoud ensure that the main.py, funcs.py and the datasets are all within the same directory before running this script.

## Python Environment
To ensure reproducibility of our analysis, the [env_concurrent_extremes.yml](https://github.com/VUB-HYDR/Mapping_cooccurring_climate_extremes/blob/3e59d679c6e31ce707907f5330b859dcff86a44e/env_concurrent_extremes.yml) provides a clone of our python environment, with all of its packages and versions. Users should use their terminal or an Anaconda Prompt to create their environment using this env_concurrent_extremes_in_east_africa.yml.

## Grid cell area
[Entire_globe_grid_cell_area.nc](https://github.com/VUB-HYDR/concurrent_climate_extremes_in_east_africa/blob/9225c1989e3bad7f7d121c4a5504c2299226c885/entire_globe_grid_cell_areas.nc) is used in this analysis for the grid cell areas. 

## Authors
Gabriele Messori

Derrick Muheki

Emanuele Bevacqua

Wim Thiery
