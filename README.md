# Welcome! 
This repository contains notebooks that test some basic models within several out-of-memory data frameworks in Python, and measures the time it takes to run them. A few notes on how to get started:
1. As the data is too large to include in this repository, first go ahead and download the original dataset from [Kaggle](/https://www.kaggle.com/datasets/erikbiswas/higgs-uci-dataset), and save it in the Data folder.
2. Next, `dask_models.ipynb` contains a cell that reads this file and generates a sub- and superset of the data in Parquet format for testing purposes. A `dask_requirements.txt` has been included to create a conda environment that will run Dask. Please refer to this file on how to create the environment.
3. When the data files have been generated, the other `*_models.ipynb` notebooks can be used to run the other methods. Please note again that these require their own environments, `*_requirements.txt` files for which have also been included.
4. I hope you learn something from these notebooks - enjoy!
