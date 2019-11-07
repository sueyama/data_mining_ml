# LightGBMのモデル

import logging
import optuna
import warnings
import functools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score, classification_report
import lightgbm as lgb

random_state = 0

def LGBM_train_and_predict(X_train_all, Y_train_all)

    logging.debug('----- LightGBM Start -----')
    
    (train_x, test_x, train_y, test_y) = train_test_split(X_train_all, Y_train_all, test_size=0.2, random_state=random_state)

    logging.debug('train_x: {}' .format(train_x.shape)
    logging.debug('test_x: {}' .format(test_x.shape)
    logging.debug('train_y: {}' .format(train_y.shape)
    logging.debug('test_y: {}' .format(test_y.shape)

    def opt(X_train, Y_train, X_test, Y_test, trial):
        # optunaでのハイパーパラメータ探索
        max_depth = trial.suggest_int('max_depth', 1, 20)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
        num_leaves = trial.suggest_int('num_leaves', 10, 1000)
        n_estimators = trial.suggest_int('n_estimators', 1, 100)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
        scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 100)
        
        lightgbm_tuna = lgb.LGBMClassifier(
            random_state = random_state,
            max_depth = max_depth,
            learning_rate = learning_rate,
            num_leaves = num_leaves,
            n_estimators = n_estimators,
            min_child_weight = min_child_weight,
            scale_pos_weight = scale_pos_weight
        )
        
        lightgbm_tuna.fit(X_train, Y_train)
        tuna_pred_test = lightgbm_tuna.predict(X_test)
        
        return (1.0 - (accuracy_score(Y_test, tuna_pred_test)))
        
    study = optuna.create_study()
    study.optimize(functools.partial(opt, train_x, train_y, test_x, test_y), n_trials=150)
    
    lgbm = lgb.LGBMClassifier(**study.best_params)
    
    lgbm.fit(train_x, train_y)
    
    logging.debug('Train score: {}' .format(lgbm.score(train_x, train_y)))
    logging.debug('Test score: {}' .format(lgbm.score(test_x, test_y)))
    logging.debug('Confusion matrix:¥n {}' .format(confusion_matrix(test_y, lgbm.predict(test_x))))
    logging.debug('f1 score: {}' .format(f1_score(test_y, lgbm.predict(test_x))))
    logging.debug('Best param: {}' .format(study.best_params))
    logging.debug('Train score: {}' .format(lgbm.score(train_x, train_y)))
    logging.debug(classification_report(test_y, lgbm.predict(test_x)))

    logging.debug('----- LightGBM End -----')
    
    return lgbm