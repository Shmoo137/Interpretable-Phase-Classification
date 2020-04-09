# Interpretable phase classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3746540.svg)](https://doi.org/10.5281/zenodo.3746540)

## Influence functions for the phase transition between the Luttinger liquid (LL) and the charge density wave (CDW)
Folder "Influence_Functions_LL-CDW" contains all data and code necessary to reproduce Fig. 2 from the paper "Phase Detection with Neural Networks: Interpreting the Black Box" by A. Dawid, P. Huembeli, M. Tomza, M. Lewenstein, and A. Dauphin, namely:
- Jupyter notebook `Reproduce_Fig2.ipynb`
- `utility_general.py` and `utility_general.py` with utility functions,
- `architecture.py` specyfing the model we used,
- `data_loader.py` to load data sets from folder `datasets`,
- `influence_functions.py` containing a function to compute the gradient of the loss w.r.t to the model's parameters,
- folder `datasets` containing the original ground states with labels being the phases LL (0) or CDW (1),
- folder `model` containing the original model we used and the mask used to shuffle the training data in a way possible to follow,
- folder `influence` with the original calculated influence functions and the hessian of the original model.

All data contained in folders `datasets`, `model`, and `influence` can be reproduced in the `Reproduce_Fig2.ipynb` notebook.

Code was written by Anna Dawid (University of Warsaw & ICFO) and Patrick Huembeli (ICFO) with help of Alexandre Dauphin (ICFO)