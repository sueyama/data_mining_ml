# ランダムフォレストのモデル

import logging
import optuna
import functools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score, classification_report

random_state = 0

def RF_train_and_predict(X_train_all, Y_train_all)

    logging.debug('----- RandomForest Start -----')
    
    (train_x, test_x, train_y, test_y) = train_test_split(X_train_all, Y_train_all, test_size=0.2, random_state=random_state)

    logging.debug('train_x: {}' .format(train_x.shape)
    logging.debug('test_x: {}' .format(test_x.shape)
    logging.debug('train_y: {}' .format(train_y.shape)
    logging.debug('test_y: {}' .format(test_y.shape)

    def opt(X_train, Y_train, X_test, Y_test, trial):
        # optunaでのハイパーパラメータ探索
        max_depth = trial.suggest_int('max_depth', 1, 50)
        #max_features = trial.suggest_int('max_features', 1, len(X_train_all.columns))
        n_estimators = trial.suggest_int('n_estimators', 1, 100)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
        
        randomforestu_tuna = RandomForestClassifier(
            random_state=0,
            class_weight='balanced',
            max_depth = max_depth,
            #max_features = max_features,
            n_estimators = n_estimators,
            min_samples_leaf = min_samples_leaf,
            min_samples_split = min_samples_split
        )
        
        randomforestu_tuna.fit(X_train, Y_train)
        tuna_pred_test = randomforestu_tuna.predict(X_test)
        
        return (1.0 - (accuracy_score(Y_test, tuna_pred_test)))
        
    study = optuna.create_study()
    study.optimize(functools.partial(opt, train_x, train_y, test_x, test_y), n_trials=150)
    
    forest = RandomForestClassifier(**study.best_params, class_weight='balanced')
    
    forest.fit(train_x, train_y)
    
    logging.debug('Train score: {}' .format(forest.score(train_x, train_y)))
    logging.debug('Test score: {}' .format(forest.score(test_x, test_y)))
    logging.debug('Confusion matrix:¥n {}' .format(confusion_matrix(test_y, forest.predict(test_x))))
    logging.debug('f1 score: {}' .format(f1_score(test_y, forest.predict(test_x))))
    logging.debug('Best param: {}' .format(study.best_params))
    logging.debug('Train score: {}' .format(forest.score(train_x, train_y)))
    logging.debug(classification_report(test_y, forest.predict(test_x)))

    logging.debug('----- RandomForest End -----')
    
    return forest