# Antigravity — Implementation Plan

**Explainable ML cho Phát hiện Gian lận Thẻ Tín dụng dưới Mất cân bằng Dữ liệu**

| Trường | Giá trị |
|---|---|
| Dataset | ULB Credit Card Fraud Detection (Kaggle) |
| Phiên bản | v1.1 — 22/06/2026 |
| Tài liệu tham chiếu | Văduva et al. (2026) — CIES paper |
| Trạng thái | 🟡 Đang triển khai |

> **Lưu ý phiên bản:** v1.1 cập nhật lineup mô hình theo kết quả thực nghiệm trong CIES paper (Văduva et al., 2026):
> RF và CatBoost có CIES cao hơn XGBoost và LightGBM trên tất cả dataset và điều kiện được kiểm tra. LightGBM bị đưa xuống vị trí supplementary do CIES sụt giảm mạnh nhất khi áp SMOTE (−24pp trên HR Attrition). Các stability metric Jaccard@k / RBO / Wasserstein (v1.0) được thay thế bởi CIES làm primary metric — nhất quán với tiêu đề đề tài và thuật toán đã được validate thống kê.

---

## 1. Câu hỏi Nghiên cứu

### 1.1 RQ chính

> *Trên bài toán phát hiện gian lận giao dịch thẻ tín dụng (ULB dataset), các kỹ thuật xử lý mất cân bằng dữ liệu — **SMOTE**, **SMOTE-ENN**, và **class-weighting (no-resampling baseline)** — ảnh hưởng như thế nào đến độ ổn định của giải thích SHAP, đo bằng **CIES**, trên các mô hình Random Forest, XGBoost, CatBoost, và Logistic Regression?*

### 1.2 RQ phụ

| # | Câu hỏi |
|---|---|
| RQ1 | Kỹ thuật resampling nào gây ra mức sai lệch CIES thấp nhất trong khi duy trì PR-AUC / F1 / Recall tương đương? |
| RQ2 | Độ ổn định giải thích (CIES) có khác biệt có ý nghĩa thống kê giữa mô hình tree-based (RF, XGBoost, CatBoost) và mô hình tuyến tính (LR)? |
| RQ3 | Có thể xây dựng proof-of-concept dashboard biến CIES thành tín hiệu giám sát sống với ngưỡng cảnh báo không? |

### 1.3 Khoảng trống đã xác minh (so với CIES paper)

CIES paper chỉ test **SMOTE vs Raw** trên **credit risk / churn / HR** datasets. Thesis này đóng góp thêm:
- Thêm **SMOTE-ENN** (hybrid technique — chưa được test trong paper)
- Domain **credit card transaction fraud** với PCA-anonymized features (V1–V28)
- So sánh cross-family: tree-based vs. linear

---

## 2. Thiết kế Thí nghiệm

### 2.1 Lineup Mô hình

| Model | Vai trò | Lý do chọn | CIES dự kiến* |
|---|---|---|---|
| **Random Forest** | Primary | CIES cao nhất mọi dataset trong paper (0.77–0.97); bagged ensemble cho smooth decision boundary | Cao |
| **XGBoost** | Primary | SOTA fraud detection; PR-AUC tốt nhất; TreeSHAP exact | Trung bình |
| **CatBoost** | Primary | CIES tốt nhất trong gradient-boosted family (0.87–0.97); balance tốt accuracy–credibility | Cao |
| **Logistic Regression** | Baseline linear | Kiểm tra tính phổ quát cross-family; SHAP linear explainer đơn giản | TBD |
| ~~LightGBM~~ | ~~Supplementary~~ | Volatile nhất dưới SMOTE (−24pp CIES trên HR Attrition); leaf-wise growth nhạy cảm với synthetic data | ~~Thấp~~ |

> *Dự kiến dựa trên kết quả Văduva et al. (2026) — có thể khác trên ULB fraud dataset do đặc thù PCA features.

**LightGBM**: Có thể thêm vào Appendix nếu muốn bổ sung so sánh, không nên là mô hình chính.

### 2.2 Điều kiện Imbalance

| Condition | Kỹ thuật | Class Ratio (xấp xỉ) | Ghi chú |
|---|---|---|---|
| **C0 — Baseline** | Class-weighting (`scale_pos_weight`) | Raw (~577:1) | Không tạo synthetic data; áp dụng vào loss function |
| **C1 — SMOTE** | `SMOTE(random_state=42)` | 1:1 (sau resample) | Pure oversampling |
| **C2 — SMOTE-ENN** | `SMOTEENN(random_state=42)` | ~1:3–1:5 (tùy ENN) | Hybrid: oversample + clean noise |

> **ADASYN**: Bỏ ra khỏi scope chính để giữ thiết kế nhất quán với paper gốc. Có thể thêm trong Future Work.

### 2.3 Ma trận Điều kiện (12 cấu hình)

|  | C0 (Class-weight) | C1 (SMOTE) | C2 (SMOTE-ENN) |
|---|:---:|:---:|:---:|
| Random Forest | ✓ | ✓ | ✓ |
| XGBoost | ✓ | ✓ | ✓ |
| CatBoost | ✓ | ✓ | ✓ |
| Logistic Regression | ✓ | ✓ | ✓ |

Mỗi cấu hình đánh giá CIES trên **N = 100** test instances (random sample), **K = 20** perturbed neighbors, **ε = 0.03**.

### 2.4 Tham số CIES (theo paper)

| Tham số | Giá trị | Mô tả |
|---|---|---|
| N | 100 | Số test instances để tính CIES model-level |
| K | 20 | Số perturbed neighbors mỗi instance |
| ε (default) | 0.03 | Noise level (3% multiplicative Gaussian) |
| ε (sensitivity) | {0.01, 0.03, 0.05, 0.10} | Dùng cho sensitivity analysis |

---

## 3. Pipeline Thực thi

### Phase 1 — Data & EDA

```
Input : creditcard.csv (284,807 rows × 30 features + 1 target)
Output: train_raw.pkl, test.pkl (cố định, không bao giờ resample)
```

**Việc cần làm:**
- Kiểm tra missing values, phân phối class (fraud ~0.17%)
- Phân tích `Amount` và `Time` (V1–V28 đã PCA — không cần hiểu nghĩa)
- Stratified split **80/20** (`random_state=42`) → `X_train`, `X_test`, `y_train`, `y_test`
- Test set được **khóa cố định** từ đây — không có bất kỳ transformation nào từ training data

> ⚠️ **Data leakage checkpoint**: Scaler chỉ được `fit` trên `X_train`, sau đó `transform` cả train và test. Không fit trên toàn bộ dataset.

### Phase 2 — Preprocessing

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train[['Amount', 'Time']])
X_test_scaled  = scaler.transform(X_test[['Amount', 'Time']])
# V1–V28: giữ nguyên (đã chuẩn hóa qua PCA)
```

- Dùng `RobustScaler` (ít nhạy outlier hơn `StandardScaler`)
- Không scale V1–V28
- Lưu scaler bằng `joblib.dump()` để reproducibility

### Phase 3 — Imbalance Handling

```python
# Chỉ áp dụng trên X_train — KHÔNG bao giờ chạm X_test
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# C0: class-weighting — không resample, truyền vào model parameter
# C1: SMOTE
X_c1, y_c1 = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)

# C2: SMOTE-ENN
X_c2, y_c2 = SMOTEENN(random_state=42).fit_resample(X_train_scaled, y_train)
```

Log lại ratio sau resample để report.

### Phase 4 — Model Training

Hyperparameters theo thiết lập của CIES paper (Section 3.8) để đảm bảo reproducibility:

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# C0 (class-weighting) — ví dụ với XGBoost
ratio = y_train.value_counts()[0] / y_train.value_counts()[1]  # ~577

models = {
    'RF': RandomForestClassifier(
        n_estimators=200, min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced',  # chỉ dùng cho C0
        random_state=42, n_jobs=-1
    ),
    'XGB': XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=ratio,  # chỉ dùng cho C0
        random_state=42, eval_metric='aucpr'
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        auto_class_weights='Balanced',  # chỉ dùng cho C0
        random_seed=42, verbose=0
    ),
    'LR': LogisticRegression(
        class_weight='balanced',  # chỉ dùng cho C0
        max_iter=1000, random_state=42
    )
}
```

> **C1 và C2**: Dùng model không có `class_weight` / `scale_pos_weight` (vì data đã balanced qua resampling).

### Phase 5 — Đánh giá Hiệu suất

```python
from sklearn.metrics import (
    average_precision_score, f1_score,
    recall_score, roc_auc_score, confusion_matrix
)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'PR-AUC'  : average_precision_score(y_test, y_prob),   # PRIMARY
        'ROC-AUC' : roc_auc_score(y_test, y_prob),
        'F1'      : f1_score(y_test, y_pred),
        'Recall'  : recall_score(y_test, y_pred),
        'CM'      : confusion_matrix(y_test, y_pred)
    }
```

> **Threshold**: Dùng precision-recall curve để tìm threshold tối ưu F1, không cứng ở 0.5.

### Phase 6 — SHAP Computation

```python
import shap

def get_shap_explainer(model, X_train, model_type='tree'):
    if model_type == 'tree':
        return shap.TreeExplainer(model)       # exact Shapley — RF, XGB, CatBoost
    elif model_type == 'linear':
        return shap.LinearExplainer(model, X_train)  # LR

# Tính SHAP values trên cùng X_test cố định cho tất cả cấu hình
shap_values = explainer.shap_values(X_test_sample)
# Với binary classification: lấy class 1
if isinstance(shap_values, list):
    shap_values = shap_values[1]
```

> ⚠️ **PCA note**: V1–V28 không có domain meaning. SHAP analysis chỉ phục vụ mục đích **kỹ thuật** (đo stability) chứ không phải **domain insight**. Cần làm rõ điều này trong thesis.

### Phase 7 — CIES Computation

Xem Section 4 (CIES Implementation) bên dưới.

### Phase 8 — Phân tích & Báo cáo

- Bảng tổng hợp: `mean CIES ± std` cho tất cả 12 cấu hình (dạng Table 1 trong CIES paper)
- Boxplot phân phối CIES instance-level (dạng Figure 5 trong paper)
- Scatter plot PR-AUC vs CIES — visualize accuracy–credibility trade-off (Figure 3)
- Sensitivity analysis CIES theo ε ∈ {0.01, 0.03, 0.05, 0.10}
- Wilcoxon signed-rank test: CIES (rank-weighted) vs Uniform baseline — tất cả 12 cấu hình

---

## 4. CIES Implementation

### 4.1 Công thức (Văduva et al., 2026)

**Bước 1 — Harmonic weights** (penalize instability ở top features nặng hơn):

$$w_j = \frac{1/r_j}{\sum_{i=1}^{M} 1/r_i}, \quad r_j = \text{rank}(|\phi_j(x)|) \text{ giảm dần}$$

**Bước 2 — Rank-weighted distance** (giữa explanation gốc và explanation trên perturbed neighbor):

$$D_R(\phi(x), \phi(x'_k)) = \sum_{j=1}^{M} w_j \cdot |\phi_j(x) - \phi_j(x'_k)|$$

**Bước 3 — CIES instance-level**:

$$\text{CIES}(x) = \max\!\left(0,\; 1 - \frac{\bar{D}_R}{\|\phi(x)\|_w}\right)$$

Trong đó $\bar{D}_R = \frac{1}{K}\sum_{k=1}^{K} D_R$ và $\|\phi(x)\|_w = \sum_j w_j |\phi_j(x)|$.

**Bước 4 — Model-level aggregation**:

$$\overline{\text{CIES}}_f = \frac{1}{N}\sum_{i=1}^{N} \text{CIES}(x_i) \pm \text{SD}$$

Score ∈ [0, 1] — giá trị 1 là perfectly stable, gần 0 là fragile/không đáng tin.

### 4.2 Python Implementation

```python
import numpy as np
import shap
from scipy.stats import wilcoxon


class CIESEvaluator:
    """
    Credibility Index via Explanation Stability
    Theo Algorithm 1, Văduva et al. (2026)
    """

    def __init__(
        self,
        model,
        model_type: str = 'tree',   # 'tree' | 'linear'
        X_background=None,           # Cần cho linear explainer
        noise_level: float = 0.03,   # epsilon
        n_neighbors: int = 20,       # K
        n_instances: int = 100,      # N
        random_state: int = 42
    ):
        self.noise_level = noise_level
        self.n_neighbors = n_neighbors
        self.n_instances = n_instances
        np.random.seed(random_state)

        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            assert X_background is not None
            self.explainer = shap.LinearExplainer(model, X_background)
        else:
            raise ValueError("model_type phải là 'tree' hoặc 'linear'")

    # ------------------------------------------------------------------
    def _get_shap(self, X):
        """Trả về SHAP values cho class 1 (binary classification)."""
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        return np.atleast_2d(sv)

    # ------------------------------------------------------------------
    def _harmonic_weights(self, phi: np.ndarray) -> np.ndarray:
        """Tính harmonic rank weights từ SHAP vector của một instance."""
        n = len(phi)
        # rank 1 = most important (rank theo |phi| giảm dần)
        ranks = np.argsort(np.argsort(-np.abs(phi))) + 1
        inv_ranks = 1.0 / ranks
        return inv_ranks / inv_ranks.sum()

    # ------------------------------------------------------------------
    def _perturb(self, x: np.ndarray) -> np.ndarray:
        """Gaussian noise tỉ lệ với |x_j|, categorical features giữ nguyên."""
        sigmas = np.where(x != 0, self.noise_level * np.abs(x), self.noise_level)
        return x + np.random.normal(0, sigmas)

    # ------------------------------------------------------------------
    def _instance_cies(self, x: np.ndarray, phi_x: np.ndarray) -> tuple[float, float]:
        """
        Tính CIES(x) và Baseline(x) (uniform weights) cho một instance.
        Returns: (cies_score, baseline_score)
        """
        M = len(phi_x)
        weights_R = self._harmonic_weights(phi_x)           # rank-weighted
        weights_U = np.full(M, 1.0 / M)                     # uniform baseline

        mag_R = np.sum(weights_R * np.abs(phi_x))
        mag_U = np.sum(np.abs(phi_x))                        # = sum |phi| / M * M

        if mag_R == 0:
            return 1.0, 1.0

        d_R_list, d_U_list = [], []

        for _ in range(self.n_neighbors):
            x_k  = self._perturb(x)
            phi_k = self._get_shap(x_k.reshape(1, -1)).flatten()

            diff = np.abs(phi_x - phi_k)
            d_R_list.append(np.dot(weights_R, diff))
            d_U_list.append(np.dot(weights_U, diff))

        d_R_mean = np.mean(d_R_list)
        d_U_mean = np.mean(d_U_list)

        cies     = max(0.0, 1 - d_R_mean / mag_R)
        baseline = max(0.0, 1 - d_U_mean * M / max(mag_U, 1e-10))
        return cies, baseline

    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray) -> dict:
        """
        Tính CIES cho N instances ngẫu nhiên từ test set.
        Returns dict với mean, std, phân vị, và raw scores.
        """
        n = min(self.n_instances, len(X_test))
        idx = np.random.choice(len(X_test), n, replace=False)
        X_sample = X_test[idx]

        # Batch SHAP (nhanh hơn gọi từng instance)
        phi_all = self._get_shap(X_sample)

        cies_scores, base_scores = [], []
        for i in range(n):
            c, b = self._instance_cies(X_sample[i], phi_all[i])
            cies_scores.append(c)
            base_scores.append(b)

        cs = np.array(cies_scores)
        bs = np.array(base_scores)

        # Wilcoxon test: CIES vs Baseline (paired, two-sided)
        stat, pval = wilcoxon(cs, bs)

        return {
            'cies_mean'   : cs.mean(),
            'cies_std'    : cs.std(),
            'cies_min'    : cs.min(),
            'cies_q25'    : np.percentile(cs, 25),
            'cies_median' : np.median(cs),
            'cies_q75'    : np.percentile(cs, 75),
            'cies_max'    : cs.max(),
            'baseline_mean': bs.mean(),
            'wilcoxon_stat': stat,
            'wilcoxon_p'  : pval,
            'significant' : pval < 0.01,
            'cies_scores' : cs.tolist(),
            'baseline_scores': bs.tolist(),
            'n_instances' : n
        }

    # ------------------------------------------------------------------
    def sensitivity_analysis(self, X_test: np.ndarray,
                              epsilons=(0.01, 0.03, 0.05, 0.10)) -> dict:
        """Chạy CIES trên nhiều noise level để validate robustness."""
        results = {}
        original_eps = self.noise_level
        for eps in epsilons:
            self.noise_level = eps
            res = self.evaluate(X_test)
            results[eps] = res['cies_mean']
        self.noise_level = original_eps
        return results
```

### 4.3 Usage Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- Load & split ---
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1).values
y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Train một mô hình (ví dụ XGBoost + C1 SMOTE) ---
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

X_tr_sm, y_tr_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5,
                      subsample=0.8, colsample_bytree=0.8,
                      random_state=42, eval_metric='aucpr')
model.fit(X_tr_sm, y_tr_sm)

# --- Đánh giá CIES ---
evaluator = CIESEvaluator(model, model_type='tree',
                          noise_level=0.03, n_neighbors=20, n_instances=100)
result = evaluator.evaluate(X_test)
print(f"CIES: {result['cies_mean']:.3f} ± {result['cies_std']:.3f}")
print(f"Baseline: {result['baseline_mean']:.3f}")
print(f"Wilcoxon p-value: {result['wilcoxon_p']:.2e} {'✓ significant' if result['significant'] else '✗'}")

# --- Sensitivity analysis ---
sa = evaluator.sensitivity_analysis(X_test)
for eps, score in sa.items():
    print(f"  ε={eps}: CIES = {score:.3f}")
```

---

## 5. Metrics & Báo cáo

### 5.1 Performance metrics (Phase 5)

| Metric | Mục đích | Priority |
|---|---|---|
| **PR-AUC** | Đánh giá tổng thể trên imbalanced data | 🔴 Primary |
| **Recall** | Tỷ lệ phát hiện fraud — quan trọng về nghiệp vụ | 🔴 Primary |
| **F1** | Cân bằng Precision-Recall | 🟡 Secondary |
| ROC-AUC | Reference only — bị inflate với imbalanced data | ⚪ Reference |

### 5.2 Stability metric (Phase 7)

| Metric | Mục đích | Priority |
|---|---|---|
| **CIES mean ± std** | Stability tổng thể của mô hình | 🔴 Primary |
| **CIES distribution** | Phát hiện edge-case instances có CIES thấp | 🟡 Secondary |
| **Baseline (uniform)** | Đối chiếu để validate rank-weighting | 🟡 Secondary |
| **Wilcoxon p-value** | Kiểm định thống kê CIES vs Baseline | 🟡 Secondary |
| **Sensitivity (ε)** | Validate robustness của CIES metric | 🟡 Secondary |

### 5.3 Kết quả tổng hợp (template Table)

| Dataset | Model | Condition | PR-AUC | F1 | Recall | CIES | Baseline | p-value |
|---|---|---|---|---|---|---|---|---|
| ULB Fraud | RF | C0 | — | — | — | — ± — | — | — |
| ULB Fraud | RF | C1 (SMOTE) | — | — | — | — ± — | — | — |
| ULB Fraud | RF | C2 (SMOTE-ENN) | — | — | — | — ± — | — | — |
| ... | XGBoost | ... | | | | | | |
| ... | CatBoost | ... | | | | | | |
| ... | LR | ... | | | | | | |

---

## 6. Cấu trúc Project

```
antigravity/
├── data/
│   └── creditcard.csv              # Raw data (không commit lên Git)
│
├── notebooks/
│   ├── 01_eda.ipynb                # Phase 1: EDA & phân phối
│   ├── 02_preprocessing.ipynb     # Phase 2: Scaling, split
│   ├── 03_imbalance.ipynb         # Phase 3: SMOTE / SMOTE-ENN
│   ├── 04_training.ipynb          # Phase 4: Train 12 cấu hình
│   ├── 05_performance.ipynb       # Phase 5: PR-AUC, F1, Recall
│   ├── 06_shap.ipynb              # Phase 6: SHAP values
│   └── 07_cies.ipynb              # Phase 7: CIES + sensitivity + Wilcoxon
│
├── src/
│   ├── preprocess.py              # Scaling, split utils
│   ├── imbalance.py               # SMOTE / SMOTE-ENN wrappers
│   ├── models.py                  # Model definitions + hyperparams
│   ├── evaluate.py                # PR-AUC, F1, Recall, ROC-AUC
│   ├── cies.py                    # CIESEvaluator class (Section 4.2)
│   └── visualize.py               # Plots: boxplot, scatter, sensitivity
│
├── outputs/
│   ├── models/                    # Trained model binaries (.pkl / .cbm)
│   ├── shap_values/               # Cached SHAP values (.npy)
│   ├── results/                   # JSON / CSV kết quả mỗi cấu hình
│   └── figures/                   # PNG plots
│
├── dashboard/                     # RQ3 proof-of-concept
│   └── app.py                     # Streamlit dashboard
│
├── antigravity.md                 # File này
├── requirements.txt
└── .gitignore                     # Thêm data/, outputs/models/
```

---

## 7. Dependencies

```txt
# requirements.txt
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
imbalanced-learn>=0.12
xgboost>=2.0
catboost>=1.2
shap>=0.45
scipy>=1.12
matplotlib>=3.8
seaborn>=0.13
streamlit>=1.35          # Dashboard RQ3
joblib>=1.3
```

---

## 8. Expected Results Direction

Dựa trên Văduva et al. (2026) — có thể không hoàn toàn trùng do ULB fraud có đặc thù PCA features:

| Mô hình | CIES dự kiến (Raw) | CIES dự kiến (SMOTE) | Trend |
|---|---|---|---|
| Random Forest | Cao (≥ 0.90) | Cao, ít thay đổi | Stable |
| CatBoost | Cao (≥ 0.87) | Vẫn cao, ít suy giảm | Stable |
| XGBoost | Trung bình (≥ 0.80) | Có thể suy giảm nhẹ | Moderate |
| Logistic Regression | Chưa rõ (chưa có trong paper) | ? | TBD |

**Accuracy–Credibility trade-off dự kiến:**
- XGBoost: PR-AUC cao nhất nhưng CIES không cao nhất → đây là finding thú vị
- CatBoost: balance tốt giữa hai trục → "best of both worlds"
- RF: CIES cao nhất nhưng PR-AUC có thể thấp hơn

**SMOTE-ENN vs SMOTE:**
- SMOTE-ENN làm sạch noise sau oversampling → hypothesis: CIES của SMOTE-ENN cao hơn SMOTE thuần
- Đây là điểm đóng góp chính chưa có trong CIES paper

---

## 9. Methodological Notes

### 9.1 PCA Features (V1–V28)

V1–V28 không có domain meaning do PCA anonymization. CIES analysis trên dataset này mang ý nghĩa **kỹ thuật** (đo consistency của SHAP vector), không phải **nghiệp vụ** (hiểu lý do gian lận). Cần làm rõ trong thesis:

> *"CIES trong nghiên cứu này được dùng như một metric kỹ thuật để đo tính ổn định của rank ordering trong SHAP vector, không phải để diễn giải ý nghĩa nghiệp vụ của từng feature."*

### 9.2 Test Set Leakage

Test set (`X_test`, `y_test`) phải:
- ✅ Được tạo trước tất cả preprocessing
- ✅ Giữ nguyên cho tất cả 12 cấu hình
- ✅ Chỉ được `transform` bằng scaler đã fit trên train
- ❌ Không qua bất kỳ resampling nào
- ❌ Không được dùng để fit scaler / encoder

### 9.3 Reproducibility

Luôn set `random_state=42` cho: train/test split, SMOTE, SMOTE-ENN, tất cả models, numpy seed trong CIESEvaluator.

### 9.4 CIES với Logistic Regression

LR dùng `shap.LinearExplainer` thay vì `TreeExplainer`. SHAP values từ LinearExplainer là exact nhưng có tính chất khác với TreeSHAP. Ghi chú điều này trong Discussion.

### 9.5 Wilcoxon Test

Paper dùng Wilcoxon signed-rank test (non-parametric, paired) cho N=100 instances. Đã được tích hợp trong `CIESEvaluator.evaluate()`. Mức ý nghĩa: `***` p<0.001, `**` p<0.01, `*` p<0.05.

---

## 10. Dashboard Proof-of-Concept (RQ3)

Streamlit app với 4 panels (theo thiết kế trong Khung Nghien Cuu v1.0):

| Panel | Nội dung |
|---|---|
| **Performance** | PR-AUC / F1 / Recall bar chart theo từng cấu hình |
| **Explanation Fidelity** | CIES mean ± std theo model × condition; ngưỡng cảnh báo CIES < 0.75 |
| **SHAP Comparison** | Beeswarm / waterfall cho cùng một giao dịch dưới các condition khác nhau |
| **Alert Log** | Danh sách instances có CIES < threshold — simulate production monitoring |

> Dashboard là **proof-of-concept** — không yêu cầu user study, không là đóng góp DSR đầy đủ.

---

## 11. Discussion Points 🗣️

Những điểm cần quyết định hoặc làm rõ trước khi triển khai:

1. **LightGBM**: Tôi đã để ra khỏi primary lineup. Nếu bạn muốn giữ lại để so sánh đầy đủ hơn với CIES paper (4 tree-based models), có thể thêm vào — nhưng số cấu hình tăng từ 12 lên 15, compute time tăng đáng kể.

2. **ADASYN**: Đã bỏ. Nếu muốn giữ thì cần justify rõ trong thesis tại sao dùng ADASYN (nó chưa được test trong CIES paper, nên việc include sẽ là contribution thêm chứ không có baseline để so sánh kết quả).

3. **CatBoost là mô hình chính**: Từ memory của tôi, bạn đang dùng XGBoost là best model. Tuy nhiên CIES paper cho thấy CatBoost cho CIES cao hơn rõ ràng. Bạn có muốn đặt CatBoost là primary và XGBoost là secondary, hay giữ XGBoost như cũ? Điều này ảnh hưởng đến cách framing kết quả trong thesis.

4. **N=100**: Paper dùng N=100 instances. Với ULB dataset có ~57,000 test instances (20% của 284,807), N=100 là rất conservative. Có thể tăng lên N=200–500 nếu compute cho phép, cho CI hẹp hơn.

5. **Dashboard scope**: Có cần implement thật sự hay chỉ cần screenshot mockup cho thesis? Nếu chỉ cần minh họa, có thể dùng Streamlit hoặc thậm chí HTML tĩnh để tiết kiệm thời gian.

---

*Generated with Claude (Anthropic) — tham chiếu chính: Văduva, A.-G., Oprea, S.-V., & Bâra, A. (2026). Measuring the fragility of trust: Devising Credibility Index via Explanation Stability (CIES) for business decision support systems.*
