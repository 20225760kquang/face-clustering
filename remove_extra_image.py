"""
remove_extra_image.py
---------------------
Kiểm tra thư mục pool/ và xóa ảnh không có trong ground_truth.csv
(ảnh bị kéo nhầm vào test-set – ví dụ ảnh của Ronaldo).
"""

import csv
import os

# ── Cấu hình ──────────────────────────────────────────────────────────────────
TESTSET_DIR = "dataset/test_set"
GT_CSV      = os.path.join(TESTSET_DIR, "ground_truth.csv")
POOL_DIR    = os.path.join(TESTSET_DIR, "pool")
DRY_RUN     = False   # True → chỉ in ra, không xóa thật
# ──────────────────────────────────────────────────────────────────────────────


def main():
    # 1. Đọc tập hợp filename có trong GT
    gt_filenames: set[str] = set()
    with open(GT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_filenames.add(row["pool_filename"].strip())

    print(f"[INFO] Số ảnh trong ground_truth.csv : {len(gt_filenames)}")

    # 2. Duyệt pool/ và tìm ảnh không có trong GT
    pool_files = [
        fname for fname in os.listdir(POOL_DIR)
        if os.path.isfile(os.path.join(POOL_DIR, fname))
    ]
    print(f"[INFO] Số ảnh trong thư mục pool/    : {len(pool_files)}")

    extras = [f for f in pool_files if f not in gt_filenames]

    if not extras:
        print("[OK]  Không tìm thấy ảnh dư – pool khớp hoàn toàn với GT.")
        return

    print(f"\n[FOUND] {len(extras)} ảnh không có trong GT:")
    for fname in extras:
        fpath = os.path.join(POOL_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  • {fname}  ({size_kb:.1f} KB)")

    # 3. Xóa (hoặc dry-run)
    if DRY_RUN:
        print("\n[DRY-RUN] Không xóa (DRY_RUN=True). Đặt DRY_RUN=False để xóa thật.")
        return

    confirm = input(f"\nXóa {len(extras)} ảnh trên? [y/N]: ").strip().lower()
    if confirm != "y":
        print("[ABORT] Hủy thao tác.")
        return

    for fname in extras:
        fpath = os.path.join(POOL_DIR, fname)
        os.remove(fpath)
        print(f"[DELETED] {fpath}")

    print(f"\n[DONE] Đã xóa {len(extras)} ảnh. Pool bây giờ khớp với GT.")


if __name__ == "__main__":
    main()
