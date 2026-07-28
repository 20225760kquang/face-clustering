# Face Clustering Benchmark

Bộ công cụ đánh giá hiệu năng mô hình trích xuất khuôn mặt và các thuật toán phân cụm (K-Means, DBSCAN, HDBSCAN, Infomap, Leiden).

---

## 🛠️ Cài đặt Môi trường

```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Chạy Benchmark

### Bước 1: Trích Xuất Embeddings (Phase 1)

```bash
# Baseline (InsightFace):
python benchmark/extract_embeddings.py --model_name buffalo_l --mode raw

# Model Fine-tuned (ONNX):
python benchmark/extract_embeddings.py --model_name finetuned --mode raw

# Model AdaFace:
python benchmark/extract_embeddings.py --model_name adaface --mode raw
```

### Bước 2: Đánh Giá Chi Tiết & Phân Cụm (Phase 2)

```bash
# Đánh giá tổng hợp (Verification, Identification, Clustering):
python benchmark/evaluate.py --model_name buffalo_l --mode raw

# Chạy phân cụm nhanh (Quick Cluster):
python benchmark/quick_cluster.py --model_name buffalo_l --mode raw

# Grid Search tìm siêu tham số tối ưu cho từng thuật toán:
python benchmark/grid_search_hdbscan.py --model_name buffalo_l --mode raw
python benchmark/grid_search_infomap.py --model_name buffalo_l --mode raw
python benchmark/grid_search_leiden.py --model_name buffalo_l --mode raw
```

*(Kết quả báo cáo Markdown/JSON và đồ thị ROC, CMC, t-SNE sẽ tự động được lưu tại `benchmark/results/`)*
