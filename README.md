data_science
==============================

data_mining_ml

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- コマンド実行用のファイル (例)`make data` `make train`
    ├── README.md          
    │
    ├── config             <- 特徴量の制限など、コンフィグファイル.
    │   └── default.json   
    │
    ├── data
    │   ├── external       <- 外部、サードパーティのソースからのデータ.
    │   ├── interim        <- 暫定、加工された中間データ.
    │   ├── processed      <- 処理済み、モデリング用の最終的な標準データセット.
    │   └── raw            <- 元の不変のデータダンプ.
    │
    ├── logs               <- ログデータを残すフォルダ.
    │
    ├── docs               <- デフォルトのSphinxプロジェクト. 詳細については、sphinx-doc.orgを参照.
    │
    ├── models             <- トレーニングおよびシリアル化されたモデル、モデル予測、またはモデルサマリー.
    │
    ├── notebooks          <- Jupyterノートブック。 命名規則には採番すること.
    │                         (例)　`initial_data_science_1.0`.
    │
    ├── references         <- データ辞書、マニュアル、その他すべての説明資料.
    │
    ├── reports            <- レポート用. HTML、PDF、LaTeXなどとして生成された分析を格納する.
    │   └── figures        <- レポートで使用される生成されたグラフィックと図
    │
    ├── requirements.txt   <- 分析環境を再現するための要件ファイル, 
    │                         インストールが必要なモジュール. setup.pyの実行にてインストールされる.
    │
    ├── setup.py           <- プロジェクトのpipをインストール可能にする (pip install -e .)
    ├── src                <- このプロジェクトで使用するソースコード.
    │   ├── __init__.py    <- srcをPythonモジュールにする. 
    │   │
    │   ├── data           <- データをダウンロードまたは生成するスクリプト.
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- 生データをモデリング用の機能に変換するスクリプト.
    │   │   └── build_features.py
    │   │
    │   ├── models         <- モデルをトレーニングし、トレーニングされたモデルを使用して作成するスクリプト
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- 結果の視覚化を作成するスクリプト
    │       └── visualize.py
    │
    └── tox.ini            <- toxを実行するための設定を含むtoxファイル. tox.testrun.orgを参照.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
