# 30 Days of ML

[30 Days of ML](https://www.kaggle.com/thirty-days-of-ml) was an event ran by Kaggle with the purpose of getting novice data scientists into Machine Learning and Kaggle competitions.

According to the competition overview, "the dataset used for this competition is synthetic (and generated using a CTGAN), but based on a real dataset. The original dataset deals with predicting the amount of an insurance claim. Although the features are anonymized, they have properties relating to real-world features."

## The Data

From the competition description: "the feature columns, `cat0 - cat9` are categorical, and the feature columns `cont0 - cont13` are continuous." The continuous valued `target` represents the variable we wish to predict in the test set. We do not include the competition data files with this repository to save space, please download the [competition files](https://www.kaggle.com/c/30-days-of-ml/data) and extract them into `~\data` before running any of these scripts.

## The Repo

The majority of the work I did for this 15 day competition was done using Kaggle notebooks. I tested out several GPU-enabled XGBoost models, however my original preprocessing and cross-validation schemes were leaking validation data into my models and/or training data. By this point, I had used up most of my weekly GPU accelerator quota and XGBoost ran far too slowly with CPU so I switched over to using LightGBM models. It turned out to give me better results and I could easily and efficiently run hyperparameter searches on my local machine.

This repository contains my work relating to 30 days of ML which I ran locally rather than on Kaggle notebooks. I did eventually revisit XGBoost doing hyperparameter searches on Kaggle with GPU enabled but doing my final fitting and predicting using the best found parameters on CPU.

My final submission involved stacking 4 of my best performing LightGBM models with 2 of my best XGBoost models which is outlined in the `Model Stacking.ipynb` notebook. Unfortunately, I ran out of time and didn't get to explore stacking as much as I would have liked to so my first stacked model was also my last submission but this was still a good learning experience.

* `~/output` - pickled output from optuna and XGBoost predictions
* `~/submissions` - final predictions in the correct format
* `lightgbm_search.py` - script for doing a hyperparameter search using optuna
* `Model Stacking.ipynb` - notebook for stacking XGBoost and LightGBM models
* `Optuna Visualizations.ipynb` - notebook for evaluating optuna results
