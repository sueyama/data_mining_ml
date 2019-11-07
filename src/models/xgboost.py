# XGBoostのモデル

import logging
import optuna
import warnings
import functools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score, classification_report
import xgboost as xgbt

random_state = 0

def XGB_train_and_predict(X_train_all, Y_train_all)

    logging.debug('----- XGBoost Start -----')
    
    (train_x, test_x, train_y, test_y) = train_test_split(X_train_all, Y_train_all, test_size=0.2, random_state=random_state)

    logging.debug('train_x: {}' .format(train_x.shape)
    logging.debug('test_x: {}' .format(test_x.shape)
    logging.debug('train_y: {}' .format(train_y.shape)
    logging.debug('test_y: {}' .format(test_y.shape)

    def opt(X_train, Y_train, X_test, Y_test, trial):
        # optunaでのハイパーパラメータ探索
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
        n_estimators = trial.suggest_int('n_estimators', 0, 1000)
        scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 100)
        subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
        colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)
        
        xgboost_tuna = xgbt.XGBClassifier(
            random_state = random_state,
            max_depth = max_depth,
            min_child_weight = min_child_weight,
            n_estimators = n_estimators,
            scale_pos_weight = scale_pos_weight,
            subsample = subsample,
            colsample_bytree = colsample_bytree
        )
        
        xgboost_tuna.fit(X_train, Y_train)
        tuna_pred_test = xgboost_tuna.predict(X_test)
        
        return (1.0 - (accuracy_score(Y_test, tuna_pred_test)))
        
    study = optuna.create_study()
    study.optimize(functools.partial(opt, train_x, train_y, test_x, test_y), n_trials=150)
    
    xgb = xgbt.LGBMClassifier(**study.best_params)
    
    lgbm.fit(train_x, train_y)
    
    logging.debug('Train score: {}' .format(xgb.score(train_x, train_y)))
    logging.debug('Test score: {}' .format(xgb.score(test_x, test_y)))
    logging.debug('Confusion matrix:¥n {}' .format(confusion_matrix(test_y, xgb.predict(test_x))))
    logging.debug('f1 score: {}' .format(f1_score(test_y, xgb.predict(test_x))))
    logging.debug('Best param: {}' .format(study.best_params))
    logging.debug('Train score: {}' .format(xgb.score(train_x, train_y)))
    logging.debug(classification_report(test_y, xgb.predict(test_x)))

    logging.debug('----- XGBoost End -----')
    
    return xgb