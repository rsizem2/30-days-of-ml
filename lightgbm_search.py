#### GLOBAL VARIABLES ####
FOLD_SEED = 3027
NUM_FOLDS = 5
VALID_FOLDS = 5
EARLY_STOP = 200
TRIALS = 100
SAVE = True
SUBMIT = True


# IMPORTS
import os
import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from random import sample
from lightgbm import LGBMRegressor
import optuna
import pickle
import time
import warnings

# Mute warnings
warnings.filterwarnings('ignore')

# Load the training data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Define Folds
train["kfold"] = -1
kf = KFold(NUM_FOLDS, shuffle = True, random_state = FOLD_SEED) 
for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    train.loc[valid_idx,"kfold"] = fold

# List of categorical/numerical columns
object_cols = [col for col in train.columns if 'cat' in col]
number_cols = [col for col in train.columns if 'cont' in col]
columns = number_cols + object_cols

def score(model_params = {}, 
          fit_params= {}, 
          verbose = True):
    
    X = train.copy()
    assert VALID_FOLDS <= NUM_FOLDS
    scores = np.zeros(VALID_FOLDS)
    if verbose: print("\n"+str(VALID_FOLDS)+" Folds:", end = ' ')
    for i,j in enumerate(sample(range(NUM_FOLDS),k = VALID_FOLDS)):
        X_train = X[X.kfold != j][columns].copy()
        X_valid = X[X.kfold == j][columns].copy()
        y_train = X[X.kfold != j]['target'].copy()
        y_valid = X[X.kfold == j]['target'].copy()
            
        encoder = OrdinalEncoder(cols = object_cols)
        X_train = encoder.fit_transform(X_train)
        X_valid = encoder.transform(X_valid)

        model = LGBMRegressor(**{**{'random_state': 0, 
                                    'n_jobs': -1,
                                   },
                                **model_params})
        model.fit(X_train, y_train,
                  verbose=1500,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric="rmse",
                  categorical_feature = object_cols,
                  early_stopping_rounds = EARLY_STOP
                  )
        
        preds = model.predict(X_valid)
        scores[i] = mean_squared_error(y_valid, preds, squared=False)
        if verbose: print(j+1, end = '')
    if verbose:
        print("\nAverage (RMSE):", scores.mean())
        print("Worst (RMSE):", scores.max())
    return round(scores.max(), 6)
    

# Optuna
def param_search(trials):
    def objective(trial):
        model_params = {
            'extra_trees': trial.suggest_categorical('extra_trees',[True,False]),
            'path_smooth': trial.suggest_float('path_smooth', 0.0, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'n_estimators': 30000,
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'num_leaves': trial.suggest_int('num_leaves', 4, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 30),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),  
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 100),
            'max_bin':trial.suggest_int('max_bin', 200, 1500),
            'cat_smooth':trial.suggest_float('cat_smooth', 0, 100),
            'cat_l2':trial.suggest_float('cat_l2', 0, 100),
    }
        return score(model_params = model_params, verbose = False)
    
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(direction="minimize")
    
    # retrieve and enqueue old parameters
    for dirname, _, filenames in os.walk('output'):
        for i, filename in enumerate(reversed(filenames)):
            if i >= trials: continue
            old_study = pickle.load(open(os.path.join(dirname, filename), "rb"))
            study.enqueue_trial(old_study.best_params)
    study.optimize(objective, n_trials=trials)
    return study

study = param_search(TRIALS)
print("\nBest Values:",study.best_params)

#print(study.trials_dataframe().sort_values('value',axis=0).head(10))

# Save study
if SAVE:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pickle.dump(study, open("output/study_lgbm_"+timestr+".p","wb"))

def create_submission(params, submit = False):
    X = train.copy()
    folds = NUM_FOLDS

    predictions =  np.zeros((test.shape[0],))
    total_rmse = 0
    max_rmse = 0
    model_params = {'random_state': 0, 
                    'n_jobs': -1,
                    'n_estimators': 20000,}

    for i in range(folds):
        X_train = X[X.kfold != i][columns].copy()
        X_valid = X[X.kfold == i][columns].copy()
        y_train = X[X.kfold != i]['target'].copy()
        y_valid = X[X.kfold == i]['target'].copy()
        X_test = test.set_index('id')[columns]

        # Label Encode Data
        encoder = OrdinalEncoder(cols = object_cols)
        X_train = encoder.fit_transform(X_train)
        X_valid = encoder.transform(X_valid)
        X_test = encoder.transform(X_test)

        model = LGBMRegressor(**{**model_params, **params})
        model.fit(X_train, y_train,
                  verbose=1500,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric="rmse",
                  categorical_feature = object_cols,
                  early_stopping_rounds = EARLY_STOP,
                  )

        predictions += model.predict(X_test) / folds 
        preds_valid = model.predict(X_valid)

        fold_rmse = mean_squared_error(y_valid, preds_valid, squared=False)
        if fold_rmse > max_rmse: 
            max_rmse = fold_rmse
        total_rmse += fold_rmse / folds
        print("Fold "+str(i)+" (RSME):", fold_rmse)

    print("Average (RMSE):", total_rmse)
    print("Worst (RMSE):", max_rmse)
    if submit:
        output = pd.DataFrame({'Id': X_test.index,'target': predictions})
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output.to_csv('submissions/submission_lgbm_'+timestr+'.csv', index=False)
    
create_submission(study.best_params, submit = SUBMIT)