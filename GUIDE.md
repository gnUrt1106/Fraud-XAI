# Hướng dẫn Chạy Dự án: CIES Explainable Fraud Detection

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường, chạy hyperparameter tuning (Optuna), chạy thí nghiệm chính (SHAP & CIES), và trực quan hóa kết quả trên Streamlit Dashboard.

---

## 1. Thiết lập Môi trường (Local)
Dự án yêu cầu **Python 3.12** để tránh các lỗi không tương thích phiên bản.

```bash
# Tạo môi trường ảo
python3.12 -m venv .venv

# Kích hoạt môi trường ảo
source .venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 2. Tune Hyperparameters bằng Optuna (Kaggle)
Để tránh quá tải máy cá nhân, hãy chạy tuning trên Kaggle. Clone code mới nhất và chia thành 2 phiên chạy (Notebooks) khác nhau để tối ưu quota GPU:

* **Phiên GPU (Chọn GPU P100 hoặc T4x2 làm Accelerator):**
  Dùng để tune các mô hình boosting hạng nặng: `XGBClassifier` và `CatBoostClassifier`.
  ```bash
  !git clone https://github.com/gnUrt1106/Fraud-XAI.git
  %cd Fraud-XAI
  !pip install -r requirements.txt
  !python tune_hyperparams.py --model gpu --trials 50 --patience 10
  ```
  *(Các mô hình sẽ được tăng tốc bằng GPU CUDA. Lệnh sẽ tự động dừng nếu sau 10 trials liên tiếp không cải thiện PR-AUC).*

* **Phiên CPU (Không tốn GPU quota):**
  Dùng để tune các mô hình chạy CPU hiệu quả: `RandomForestClassifier` và `LogisticRegression`.
  ```bash
  !git clone https://github.com/gnUrt1106/Fraud-XAI.git
  %cd Fraud-XAI
  !pip install -r requirements.txt
  !python tune_hyperparams.py --model cpu --trials 30 --patience 5
  ```
  *(RandomForest sẽ tự động sử dụng song song tất cả các nhân CPU có sẵn trên Kaggle).*

---

## 3. Chạy Thí nghiệm để lấy SHAP Value & CIES

Sau khi nhận được các tham số tối ưu (được cập nhật tự động hoặc thủ công trong `configs/default.yaml` và `src/models.py`), bạn hãy chạy script thí nghiệm chính để tạo toàn bộ dữ liệu phân tích.

```bash
# Chạy toàn bộ thí nghiệm (4 mô hình × 3 điều kiện mất cân bằng)
# Tạo SHAP values, tính CIES index, sinh heatmap & biểu đồ so sánh:
python run_experiments.py
```

* **Lưu ý:** 
  - Nếu muốn chạy nhanh để xem kết quả Performance trước mà không tính CIES (vì tính CIES yêu cầu perturb 100 instance × 20 neighbor), bạn có thể chạy: `python run_experiments.py --skip-cies`.
  - Kết quả SHAP values của tập test sẽ được lưu trữ trong thư mục `outputs/shap_values/`.
  - Kết quả metrics và CIES score chi tiết được lưu trữ trong `outputs/results/` dưới dạng file JSON/CSV.

---

## 4. Trực quan hóa trên Streamlit Dashboard (Local)

Mở terminal tại máy local, kích hoạt venv và khởi chạy ứng dụng Streamlit:

```bash
streamlit run dashboard/app.py
```

Ứng dụng sẽ tự động mở tại **http://localhost:8501** gồm 4 Panel:
- **Panel 1 — Performance Metrics:** So sánh PR-AUC, F1-Score, Recall giữa các điều kiện cân bằng dữ liệu khác nhau.
- **Panel 2 — Explanation Fidelity:** Theo dõi CIES stability mean, độ lệch chuẩn và cảnh báo các cấu hình có độ tin cậy giải thích thấp.
- **Panel 3 — SHAP Comparison & Instance Explorer:** 
  - Xem phân phối CIES và trade-off giữa Accuracy–Credibility.
  - **Khám phá tương tác (Interactive Instance Explorer):** Chọn một instance cụ thể từ 100 instance kiểm thử, xem CIES score và biểu đồ cột đóng góp SHAP của top 10 features quan trọng nhất đối với instance đó!
- **Panel 4 — Alert Log:** Danh sách các instance có CIES score dưới ngưỡng tin cậy để phục vụ giám sát vận hành.
