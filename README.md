# Bộ Công Cụ Benchmark Face Clustering & Model Evaluation

Hệ thống benchmark dùng để đánh giá hiệu năng các mô hình nhận diện khuôn mặt (phát hiện + căn chỉnh + trích xuất embedding) và so sánh chất lượng của các thuật toán phân cụm khuôn mặt (K-Means, DBSCAN, HDBSCAN, Agglomerative Clustering) trên tập dữ liệu kiểm thử.

---

## 📂 Ý Nghĩa Các File Trong Cây Thư Mục

```text
face-clustering/
├── config.yaml          # File cấu hình trung tâm (đường dẫn dữ liệu, mô hình, siêu tham số phân cụm)
├── face_embedder.py     # Class lõi bọc mô hình InsightFace (ONNX) và custom ONNX (fine-tuned)
├── requirements.txt     # Các thư viện Python cần cài đặt
├── dataset/
│   └── test_set/        # Thư mục chứa tập kiểm thử cố định (gồm folder pool/ và file ground_truth.csv)
├── model/
│   └── best_model.onnx  # File mô hình nhận diện đã được fine-tune (kèm best_model.onnx.data)
└── benchmark/
    ├── config.py        # Đọc cấu hình từ config.yaml và phân tích các tham số override từ CLI
    ├── clustering.py    # Chạy thử nghiệm phân cụm (Grid Search) và tính toán BCubed, Pairwise F1, ARI, NMI
    ├── make_split.py    # Script tiện ích phân chia gallery/query cố định lưu vào CSV (chạy một lần duy nhất)
    ├── extract_embeddings.py # Phase 1: Trích xuất face embeddings từ ảnh trong test_set
    ├── evaluate.py      # Phase 2: Đánh giá chi tiết Verification, Identification và Clustering Quality
    ├── embeddings/      # Thư mục đầu ra lưu file embeddings trích xuất (.npz)
    └── results/         # Thư mục đầu ra lưu báo cáo (Markdown/JSON) và đồ thị trực quan (ROC, CMC, t-SNE, UMAP)
```

---

## ⚙️ Cài đặt Môi trường
Trước khi chạy, hãy kích hoạt môi trường conda của bạn (`cv_ocr`) và cài đặt dependencies:
```bash
# Di chuyển vào thư mục face-clustering
cd src-git/face-clustering

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Chạy Benchmark

### Bước 1: Trích Xuất Embeddings (Phase 1)
Chạy script trích xuất embeddings cho mô hình bạn muốn kiểm nghiệm.

*   **Chạy với Baseline (InsightFace):**
    ```bash
    python benchmark/extract_embeddings.py --model_name buffalo_l --mode raw
    ```
*   **Chạy với Mô hình Fine-Tuned (ONNX):**
    ```bash
    python benchmark/extract_embeddings.py --model_name finetuned --mode raw
    ```
*(File embeddings thu được sẽ lưu tại `benchmark/embeddings/`)*

### Bước 2: Chạy Đánh Giá Chi Tiết (Phase 2)
Chạy đánh giá các chỉ số (ROC-AUC, TAR@FAR, EER, Rank-1/5/10, CMC, mAP, Silhouette, BCubed & Pairwise F1) và sinh đồ thị.

*   **Đánh giá Baseline:**
    ```bash
    python benchmark/evaluate.py --model_name buffalo_l --mode raw
    ```
*   **Đánh giá Mô hình Fine-Tuned:**
    ```bash
    python benchmark/evaluate.py --model_name finetuned --mode raw
    ```
*(Báo cáo kết quả và đồ thị t-SNE, UMAP, ROC, CMC sẽ được lưu tại `benchmark/results/`)*
