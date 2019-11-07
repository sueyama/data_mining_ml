# AdaBoostのモデル

import logging
import optuna
import warnings
import functools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.ensemble import AdaBoostClassifier

random_state = 0

def AD_train_and_predict(X_train_all, Y_train_all)

    logging.debug('----- AdaBoost Start -----')
    
    (train_x, test_x, train_y, test_y) = train_test_split(X_train_all, Y_train_all, test_size=0.2, random_state=random_state)

    logging.debug('train_x: {}' .format(train_x.shape)
    logging.debug('test_x: {}' .format(test_x.shape)
    logging.debug('train_y: {}' .format(train_y.shape)
    logging.debug('test_y: {}' .format(test_y.shape)

    def opt(X_train, Y_train, X_test, Y_test, trial):
        # optunaでのハイパーパラメータ探索
        n_estimators = trial.suggest_int('n_estimators', 1, 100)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.1)
        
        adaboost_tuna = AdaBoostClassifier(
            random_state = random_state,
            n_estimators = n_estimators,
            learning_rate = learning_rate
        )
        
        adaboost_tuna.fit(X_train, Y_train)
        tuna_pred_test = adaboost_tuna.predict(X_test)
        
        return (1.0 - (accuracy_score(Y_test, tuna_pred_test)))
        
    study = optuna.create_study()
    study.optimize(functools.partial(opt, train_x, train_y, test_x, test_y), n_trials=150)
    
    ada = AdaBoostClassifier(**study.best_params)
    
    ada.fit(train_x, train_y)
    
    logging.debug('Train score: {}' .format(ada.score(train_x, train_y)))
    logging.debug('Test score: {}' .format(ada.score(test_x, test_y)))
    logging.debug('Confusion matrix:¥n {}' .format(confusion_matrix(test_y, ada.predict(test_x))))
    logging.debug('f1 score: {}' .format(f1_score(test_y, ada.predict(test_x))))
    logging.debug('Best param: {}' .format(study.best_params))
    logging.debug('Train score: {}' .format(ada.score(train_x, train_y)))
    logging.debug(classification_report(test_y, ada.predict(test_x)))

    logging.debug('----- AdaBoost End -----')
    
    return ada