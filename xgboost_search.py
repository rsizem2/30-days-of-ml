#### GLOBAL VARIABLES ####
FOLD_SEED = 3027
NUM_FOLDS = 5
VALID_FOLDS = 5
EARLY_STOP = 200
TRIALS = 10
SAVE = False
SUBMIT = False


# IMPORTS
# IMPORTS
import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from random import sample
from xgboost import XGBRegressor
import optuna
import time
import pickle
import warnings

# Mute warnings
warnings.filterwarnings('ignore')

# Load the training data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# Define Folds
train["kfold"] = -1
kf = KFold(NUM_FOLDS, shuffle = True, random_state = FOLD_SEED) 
for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    train.loc[valid_idx,"kfold"] = fold

# List of categorical/numerical columns
object_cols = [col for col in train.columns if 'cat' in col]
number_cols = [col for col in train.columns if 'cont' in col]
columns = number_cols + object_cols
relevant_cols = ['cont1', 'cont5', 'cat1_A', 'cont12', 'cont13', 
                  'cont2', 'cont0', 'cat1_B', 'cat8_C', 'cont4', 
                  'cont9', 'cont10', 'cont3', 'cont11', 'cont7', 
                  'cat5', 'cont6', 'cont8', 'cat8_E']

def score(model_params = {}, 
          fit_params= {},
          remove = False,
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
            
        if remove:    
            encoder = OrdinalEncoder(cols = ['cat5','cat8'])
            X1_train = encoder.fit_transform(X_train[['cat5','cat8']])
            X1_valid = encoder. transform(X_valid[['cat5','cat8']])
            
        encoder = OneHotEncoder(cols = object_cols, use_cat_names=True)
        X_train = encoder.fit_transform(X_train)
        X_valid = encoder.transform(X_valid)
        
        if remove:
            X_train = X_train.join(X1_train)
            X_train = X_train[relevant_cols]
            X_valid = X_valid.join(X1_valid)
            X_valid = X_valid[relevant_cols]

        model = XGBRegressor(**{**{'random_state': 0,        
                                   #'booster':'gbtree',
                                   'tree_method':'hist',
                                   #'gpu_id':0,
                                   #'predictor':"gpu_predictor"
                                   },
                                **model_params})
        model.fit(X_train, y_train,
                  verbose=1500,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric="rmse",
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
            'tree_method': 'hist',
            'booster': trial.suggest_categorical('booster',['gbtree','dart']),
            'learning_rate': trial.suggest_discrete_uniform('learning_rate', 0.008, 0.05, 0.001),
            'n_estimators': 30000,
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_weight': trial.suggest_discrete_uniform('min_child_weight', 5.0, 15.0, 0.1),
            'gamma': trial.suggest_discrete_uniform('gamma', 1.0, 2.0, 0.01),
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.01, 0.25, 0.01),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.45, 0.7, 0.01),    
            'reg_lambda': trial.suggest_discrete_uniform('reg_lambda', 10.0, 30.0, 0.1),
            'reg_alpha': trial.suggest_discrete_uniform('reg_alpha', 0.0, 10.0, 0.1),
    }
        drop_shap = trial.suggest_categorical('drop_shap',[True,False])
        return score(model_params = model_params, remove = drop_shap, verbose = False)
    
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    return study

study = param_search(TRIALS)
print("\nBest Values:",study.best_params)

#print(study.trials_dataframe().sort_values('value',axis=0).head(10))

# Save study
if SAVE:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pickle.dump(study, open("../studys/study_xgboost_"+timestr+".p","wb"))

def create_submission(params, submit = False):
    X = train.copy()
    folds = NUM_FOLDS

    predictions =  np.zeros((test.shape[0],))
    total_rmse = 0
    max_rmse = 0
    model_params = {'random_state': 0, 
                    #'n_jobs': -1,
                    'n_estimators': 30000,
                    'tree_method': 'hist'}

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

        model = XGBRegressor(**{**model_params, **params})
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
        output.to_csv('../output/submission_xgboost_'+timestr+'.csv', index=False)
    
create_submission(study.best_params, submit = SUBMIT)