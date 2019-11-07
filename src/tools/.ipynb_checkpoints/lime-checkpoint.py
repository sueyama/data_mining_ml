import lime.lime_tabular
import re,itertools,json
from lime.lie_text import TextDomainMapper

def lime_predict(model, X_train_all, Y_train_all, x_test, feature_names, num_features=6,
                class_names=None, discretize_continuous=True, discretizer='quartile', categorical_features=None,
                kernel_width=None, feature_selection='auto', variable_features=None):
    
    list_lime_features = []
    
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train_all,
                                                      feature_names = feature_names,
                                                      mode = 'classification',
                                                      class_names = class_names,
                                                      discretize_continuous = discretize_continuous, #カテゴリカルデータ以外は四分位数に変換される
                                                      discretizer = discretizer, #discretize_continuousがTrueの場合にのみ。選択肢は「quarile 四分位数」「decile 十分位数」「entropy エントロピー」)
                                                       categorical_features = categorical_features, #カテゴリカルデータが含まれていれば指定
                                                       kernel_width = kernel_width, #カーネル幅。 Noneの場合、デフォルトはsqrt(列数)*0.75
                                                       feature_selection = feature_selection #特徴量選択の方法「forward_selection」「lasso_path」「none」「auto」
                                                      )
    
    exp = explainer.explain_instance(x_test[0],
                                    model.predict_proba,
                                    num_features = num_features,
                                    )
    
    list_lime_test = exp.as_lit()
    
    score_value = 0.0
    
    # LIMEの判定結果を評価する (modelとの結果が異なる場合はLIMEの値が信憑性が低いため、値を返さない)
    for item, value in list_lime_test:
        score_value += value
        
    # LIMEの結果からvariable_featuresに設定した値だけ抽出する
    if variable_features is not None:
        list_lime_test = [(item, value) for item, value in list_lime_test if item in variable_features]
        
    if score_value >= 0.0:
        if model.predict(x_test[:1]) == 2:
            list_lime_features += list_lime_test
    else:
        if model.predict(x_test[:1]) == 1:
            list_lime_features += list_lime_test
    
    return list_lime_features
