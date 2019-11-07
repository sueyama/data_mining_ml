# Catboostのモデル

import logging
import optuna
import warnings
import functools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score, classification_report
from catboost import CatBoostClassifier

random_state = 0

def CAT_train_and_predict(X_train_all, Y_train_all)

    logging.debug('----- CatBoost Start -----')
    
    (train_x, test_x, train_y, test_y) = train_test_split(X_train_all, Y_train_all, test_size=0.2, random_state=random_state)

    logging.debug('train_x: {}' .format(train_x.shape)
    logging.debug('test_x: {}' .format(test_x.shape)
    logging.debug('train_y: {}' .format(train_y.shape)
    logging.debug('test_y: {}' .format(test_y.shape)

    def opt(X_train, Y_train, X_test, Y_test, trial):
        # optunaでのハイパーパラメータ探索
        depth = trial.suggest_int('depth', 1, 10)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.1)
        l2_leaf_reg = trial.suggest_int('l2_leaf_reg', 1, 100)
        iterations = trial.suggest_int('iterations', 1, 100)
        
        catboost_tuna = CatBoostClassifier(
            random_state = random_state,
            loss_function = 'MultiClass',
            depth = depth,
            learning_rate = learning_rate,
            l2_leaf_reg = l2_leaf_reg,
            iterations = iterations
        )
        
        catboost_tuna.fit(X_train, Y_train)
        tuna_pred_test = catboost_tuna.predict(X_test)
        
        return (1.0 - (accuracy_score(Y_test, tuna_pred_test)))
        
    study = optuna.create_study()
    study.optimize(functools.partial(opt, train_x, train_y, test_x, test_y), n_trials=150)
    
    cat = CatBoostClassifier(loss_function = 'MultiClass', **study.best_params)
    
    cat.fit(train_x, train_y)
    
    logging.debug('Train score: {}' .format(cat.score(train_x, train_y)))
    logging.debug('Test score: {}' .format(cat.score(test_x, test_y)))
    logging.debug('Confusion matrix:¥n {}' .format(confusion_matrix(test_y, cat.predict(test_x))))
    logging.debug('f1 score: {}' .format(f1_score(test_y, cat.predict(test_x))))
    logging.debug('Best param: {}' .format(study.best_params))
    logging.debug('Train score: {}' .format(lgbm.score(train_x, train_y)))
    logging.debug(classification_report(test_y, cat.predict(test_x)))

    logging.debug('----- CatBoost End -----')
    
    return cat