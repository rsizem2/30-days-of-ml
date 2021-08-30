# 30 Days of ML

[30 Days of ML](https://www.kaggle.com/thirty-days-of-ml) was an event ran by Kaggle with the purpose of getting novice data scientists into Machine Learning and Kaggle competitions.

According to the competition overview, "the dataset used for this competition is synthetic (and generated using a CTGAN), but based on a real dataset. The original dataset deals with predicting the amount of an insurance claim. Although the features are anonymized, they have properties relating to real-world features."

## About this Repo

Originally, the majority of the work I did for this 15 day competition was done using Kaggle notebooks. I tested out several GPU-enabled XGBoost models, however my preprocessing and cross-validation schemes were leaking validation data into my models and/or training data.

By this point, I had used up most of my weekly GPU accelerator quota and XGBoost ran far too slowly with CPU so I switched over to using LightGBM models. It turned out to give me better results and I could easily and efficiently run hyperparameter searches on my local machine.

This repository contains my work relating to 30 days of ML which I ran locally rather than on Kaggle notebooks. I'll link any notebooks I decide to release publicly in this README after the event concludes.

## Files

* data -  original competition data files
  * sample_submission.csv - example competition submission file
  * test.csv - the test set on which our model will be evaluated
  * train.csv - the training data for our model
* output - output generated by optuna hyperparameter search
* submissions - predictions
* lightgbm_search.py - python script for hyperparameter search for LightGBM
* optuna_analysis.py - jupyter notebook evaluating the optuna output
