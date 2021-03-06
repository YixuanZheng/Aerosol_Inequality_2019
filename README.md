## Climate effects of aerosols reduce economic inequality

Replication materials for *Zheng, Davis, Persad & Caldeira (under review)*

The materials in this repository can be used to reproduce the figures and tables in the main text and supplementary information of the paper. 

If you have any questions or suggestions, please contact Yixuan Zheng (yxzheng@carnegiescience.edu).


## Scenarios
Three scenarios are considered in this study

- **With-Aerosol**: The control scenario that applies all historical anthropogenic and natural forcers
- **No-Aerosol**: The preindustrial aerosol scenario that anthropogenic aerosol emissions are fixed at 1850 levels
- **No-Sulfate**: The preindustrial sulfate scenario that anthropogenic sufalte aerosol (and its precursor SO<sub>2</sub>) emissions are fixed at 1850 levels

## Organization of repository

- **modules**: contains scripts used to replicate estimates, figures and tables appear in the paper
- **input**: contains input data
- **output**: after running scripts in **modules**, will contain all output files (data, figures, and tables)


## Instructions for replication

The repository is 226 MB. Full replication will occupy around 800 MB. 

Run Scipts 1-9 first, which will reproduce all data needed to replicate all figiures and tables. Scripts F1-4 and FS1-17 will reproduce all figures appeared in the main text and supplementary information of the paper. Script TableS1 will reproduced Table S1. 

Generated figures and tables will appear in the **output/plot** and **output/table** folders respectively. Figures and tables generated by provided scripts should differ only aesthetically from those appearing in the paper due to post-processing in Adobe Illustrator.

**Helvetica** is set as the font for all figures.

The code in this repository were written and tested in Python 3.6.6.