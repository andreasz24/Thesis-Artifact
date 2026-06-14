Corporate Credit Rating Prediction — Model Comparison

This project compares three machine learning approaches for predicting corporate credit ratings from financial ratios:


XGBoost — gradient-boosting baseline
TabPFN-3 — tabular foundation model
AutoGluon — AutoML framework


All three models are trained and evaluated on identical train/test splits so that performance differences reflect the models, not the data.


Repository structure

├── data/                      # the dataset (CSV)
├── src/                       # core Python functions
│   ├── config.py              # paths, constants, experiment grids
│   ├── data.py                # preprocessing and train/test split
│   ├── models.py              # training functions for all three models
│   ├── evaluation.py          # metrics and classification reports
│   └── plotting.py            # confusion matrices, feature importance, heatmaps
├── notebooks/                 # one notebook per model, with markdown explanations
│   ├── 1_data_exploration.ipynb
│   ├── 2_xgboost.ipynb
│   ├── 3_tabpfn.ipynb
│   └── 4_autogluon.ipynb
├── outputs/                   # generated at runtime (splits, results, figures)
├── requirements.txt
└── README.md

The src/ files contain the logic. The notebooks import from src/, run the functions, and explain each step. Hyperparameters and paths are centralised in src/config.py.


How to run

Each notebook is self-contained. Open any notebook in Google Colab and run all cells top to bottom — the first cell clones this repo and installs dependencies automatically.

NotebookWhat it doesRuntime1_data_explorationExplores the dataset and creates the train/test split~10s2_xgboostTrains and evaluates the XGBoost baseline~30s3_tabpfnRuns the TabPFN n_estimators experiment~2 min (GPU)4_autogluonRuns the AutoGluon grid search~20 min

TabPFN (notebook 3) requires a free API token from Prior Labs — instructions are in the notebook.

A GPU is recommended for notebook 3: Runtime → Change runtime type → GPU.

To run locally instead of Colab:

bashpip install -r requirements.txt
jupyter notebook


Data

The dataset contains 7,805 observations from 7 credit rating agencies. Each row is a company rated by an agency, with 16 financial ratios as features and the credit rating (D to AAA) as the target.

Source: Kaggle — Corporate Credit Rating with Financial Ratios


Reproducibility

The preprocessing and split are fully deterministic (random_state=42), so the train/test partition is identical across runs. Exact metric values may vary slightly across library versions, but the relative model comparison holds
