# Antigravity — Explainable ML for Credit Card Fraud Detection

Nghiên cứu tác động của kỹ thuật xử lý mất cân bằng dữ liệu lên độ ổn định giải thích SHAP, đo bằng **CIES** (Credibility Index via Explanation Stability).

**Dataset:** ULB Credit Card Fraud Detection (Kaggle)
**Reference:** Văduva et al. (2026) — CIES paper

## Research Questions

| # | Question |
|---|---|
| RQ1 | Kỹ thuật resampling nào gây mức sai lệch CIES thấp nhất trong khi duy trì PR-AUC / F1 / Recall tương đương? |
| RQ2 | Độ ổn định giải thích (CIES) có khác biệt thống kê giữa tree-based (RF, XGB, CatBoost) và linear (LR)? |
| RQ3 | Có thể xây dựng proof-of-concept dashboard biến CIES thành tín hiệu giám sát sống? |

## Experiment Matrix (12 configurations)

|  | C0 (Class-weight) | C1 (SMOTE) | C2 (SMOTE-ENN) |
|---|:---:|:---:|:---:|
| Random Forest | ✓ | ✓ | ✓ |
| XGBoost | ✓ | ✓ | ✓ |
| CatBoost | ✓ | ✓ | ✓ |
| Logistic Regression | ✓ | ✓ | ✓ |

## Project Structure

```
antigravity/
├── data/raw/creditcard.csv         # Raw data (not committed)
├── notebooks/
│   └── 01_eda.ipynb                # EDA & distributions
├── src/
│   ├── preprocess.py               # Load, split, RobustScaler
│   ├── imbalance.py                # SMOTE / SMOTE-ENN / class-weight
│   ├── models.py                   # Model definitions + hyperparams
│   ├── evaluate.py                 # PR-AUC, F1, Recall, ROC-AUC
│   ├── cies.py                     # CIESEvaluator class
│   ├── visualize.py                # CIES plots & comparisons
│   └── tuning/
│       └── optuna_tuner.py         # Optuna hyperparameter tuning
├── outputs/
│   ├── models/                     # Trained model binaries
│   ├── shap_values/                # Cached SHAP values
│   ├── results/                    # JSON/CSV per configuration
│   └── figures/                    # PNG plots
├── dashboard/
│   └── app.py                      # Streamlit proof-of-concept (RQ3)
├── configs/default.yaml            # Experiment configuration
├── run_experiments.py              # Run all 12 configurations
├── tune_hyperparams.py             # Optuna tuning CLI
├── antigravity.md                  # Full implementation plan
├── requirements.txt
└── .gitignore
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset:**
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
   ```

## Usage

### Run Full Experiment Matrix
```bash
python run_experiments.py
```

### Performance-only (skip CIES computation)
```bash
python run_experiments.py --skip-cies
```

### Run specific models
```bash
python run_experiments.py --models RF CatBoost
```

### Hyperparameter Tuning (Optuna)
```bash
python tune_hyperparams.py --model CatBoostClassifier --trials 50
python tune_hyperparams.py --model XGBClassifier --trials 30
python tune_hyperparams.py --model LogisticRegression --trials 20
```

### Dashboard (RQ3)
```bash
streamlit run dashboard/app.py
```

## Notebooks

- `notebooks/01_eda.ipynb` — Exploratory Data Analysis & class distribution visualization
