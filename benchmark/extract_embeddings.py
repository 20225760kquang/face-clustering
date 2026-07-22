"""
Phase 1: Extract face embeddings from dataset and save to .npz

Runs ArcFace (buffalo_l, etc.) or a fine-tuned ONNX model on GPU/CPU,
extracts 512-d embeddings for all images, splits each person into gallery/query,
and saves everything to a model-specific .npz file.

This script parses a fixed dataset with the flat pool + ground_truth.csv format.
"""

import sys
import os
import time
import cv2
import csv
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for FaceEmbedder import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_embedder import FaceEmbedder

# Import configurations from config.py

from config import cfg
# Đã đọc được tham số dòng lệnh từ khi import cfg rồi !

# Extract configurations
DATASET_DIR = cfg["dataset"]["dir_path"]
OUTPUT_DIR = cfg["extraction"]["output_path"]
GALLERY_COUNT = cfg["dataset"]["gallery_count"]
MIN_GALLERY = cfg["dataset"]["min_gallery"]
PAD_RATIO = cfg["extraction"]["pad_ratio"]
MODEL_NAME = cfg["extraction"]["model_name"]
DET_SIZE = tuple(cfg["extraction"]["det_size"])
PROVIDERS = cfg["extraction"]["providers"]
MODE = cfg["extraction"]["mode"]

EMBEDDINGS_FILE = cfg["extraction"]["embeddings_file"]
SPEED_FILE = cfg["extraction"]["speed_file"]
CUSTOM_REC_ONNX = cfg["extraction"].get("custom_rec_onnx")
ADAFACE_ARCH = cfg["extraction"].get("adaface_arch")
ADAFACE_CKPT_PATH = cfg["extraction"].get("adaface_ckpt_path")
LIMIT_PERSONS = cfg["extraction"].get("limit_persons")


def extract_embedding(embedder, img_path: str, mode: str):
    """
    Extract embedding using the chosen mode.

    - 'raw':     detect + landmark align + embed  (full pipeline)
    - 'cropped': resize + embed directly          (skip detection)

    For 'raw' mode: if face detection fails on a pre-cropped image,
    falls back to 'cropped' mode for that image.
    """
    if mode == "cropped":
        return embedder.embed_file(img_path, is_cropped=True)

    # --- raw mode: detect + align + embed ---
    results = embedder.embed_file(img_path, is_cropped=False, max_faces=1)
    if results:
        emb = results[0]["embedding"]
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb

    # Fallback: detection failed → use cropped mode
    return embedder.embed_file(img_path, is_cropped=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialize model with execution providers
    # ------------------------------------------------------------------
    print(f"Initializing FaceEmbedder — model: {MODEL_NAME}, mode: {MODE}, det_size: {DET_SIZE}")
    if MODEL_NAME == "finetuned":
        embedder = FaceEmbedder(
            model_name="buffalo_l",  # Use buffalo_l pack for RetinaFace detection/alignment
            providers=PROVIDERS,
            det_size=DET_SIZE,
            custom_rec_onnx=CUSTOM_REC_ONNX,  # Overwrite recognition with our custom ONNX model
        )
    elif MODEL_NAME == "adaface":
        embedder = FaceEmbedder(
            model_name=MODEL_NAME,
            providers=PROVIDERS,
            det_size=DET_SIZE,
            adaface_arch=ADAFACE_ARCH,
            adaface_ckpt_path=ADAFACE_CKPT_PATH,
        )
    else:
        embedder = FaceEmbedder(
            model_name=MODEL_NAME,
            providers=PROVIDERS,
            det_size=DET_SIZE,
        )

    # ------------------------------------------------------------------
    # Discover and read CSV dataset mapping
    # ------------------------------------------------------------------
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory does not exist at '{DATASET_DIR}'")
        print("Please check dataset.dir in config.yaml or pass --dataset_dir <path>")
        sys.exit(1)

    csv_path = DATASET_DIR / "ground_truth.csv"
    pool_dir = DATASET_DIR / "pool"

    if not csv_path.exists():
        print(f"Error: ground_truth.csv not found in '{DATASET_DIR}'")
        sys.exit(1)

    if not pool_dir.exists():
        print(f"Error: pool/ directory not found in '{DATASET_DIR}'")
        sys.exit(1)

    print(f"Reading ground truth annotations from: {csv_path}")
    gt_df = pd.read_csv(csv_path)

    # Validate CSV columns
    required_cols = {"pool_filename", "gt_id"}
    if not required_cols.issubset(gt_df.columns):
        print(f"Error: ground_truth.csv must contain columns: {required_cols}")
        sys.exit(1)

    # Group files by identity (gt_id)
    person_groups = gt_df.groupby("gt_id")
    print(f"Found {len(person_groups)} persons in ground truth dataset.\n")

    all_embeddings = []
    all_labels = []
    all_paths = []
    all_splits = []  # "gallery" or "query"

    total_time = 0.0
    total_images = 0
    fallback_count = 0

    # Process each identity group
    # Sort keys for deterministic processing order
    sorted_person_ids = sorted(person_groups.groups.keys())
    if LIMIT_PERSONS:
        print(f"Limiting to first {LIMIT_PERSONS} persons for testing...")
        sorted_person_ids = sorted_person_ids[:LIMIT_PERSONS]

    for person_id in sorted_person_ids:
        group_df = person_groups.get_group(person_id)

        # Get sorted images
        filenames = sorted(group_df["pool_filename"].tolist())
        images = []
        for fname in filenames:
            img_path = pool_dir / fname
            if img_path.exists():
                images.append(img_path)
            else:
                print(f"  Warning: Image {fname} mapped in CSV but not found in pool folder.")

        if not images:
            print(f"  {person_id}: no existing images — skipping")
            continue

        # Map filename to split (dynamic fallback if 'split' column is missing from CSV)
        if "split" in group_df.columns:
            filename_to_split = dict(zip(group_df["pool_filename"], group_df["split"]))
        else:
            n_imgs = len(images)
            if n_imgs > GALLERY_COUNT:
                n_gal = GALLERY_COUNT
            else:
                n_gal = min(n_imgs, MIN_GALLERY)
            
            filename_to_split = {}
            for idx, img_path in enumerate(images):
                if idx < n_gal:
                    filename_to_split[img_path.name] = "gallery"
                else:
                    filename_to_split[img_path.name] = "query"

        person_fallbacks = 0
        n_gallery = sum(1 for img in images if filename_to_split[img.name] == "gallery")
        n_query = len(images) - n_gallery

        # Extract embeddings for every image of this person
        for img_path in images:
            split = filename_to_split[img_path.name]

            start = time.perf_counter()
            try:
                if MODE == "raw":
                    # Read image and pad it — RetinaFace needs context around
                    # tight face crops to detect landmarks properly
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise FileNotFoundError(f"Cannot read: {img_path}")

                    h, w = img.shape[:2]
                    pad = int(max(h, w) * PAD_RATIO)
                    padded = cv2.copyMakeBorder(
                        img, pad, pad, pad, pad,
                        cv2.BORDER_CONSTANT, value=(128, 128, 128),
                    )

                    results = embedder.embed_raw(padded, max_faces=1)
                    if results:
                        emb = results[0]["embedding"]
                        emb = emb / (np.linalg.norm(emb) + 1e-10)
                    else:
                        # Fallback to cropped if detection fails
                        emb = embedder.embed_cropped(img)
                        person_fallbacks += 1
                        fallback_count += 1
                else:
                    emb = embedder.embed_file(str(img_path), is_cropped=True)

                elapsed = time.perf_counter() - start
                total_time += elapsed
                total_images += 1

                all_embeddings.append(emb)
                all_labels.append(person_id)
                all_paths.append(str(img_path))
                all_splits.append(split)
            except Exception as e:
                print(f"    Error on {img_path.name} ({person_id}): {e}")

        fb_info = f" ({person_fallbacks} fallback)" if person_fallbacks else ""
        print(f"  {person_id}: {n_gallery} gallery + {n_query} query = {len(images)} total{fb_info}")

    # ------------------------------------------------------------------
    # Save to .npz
    # ------------------------------------------------------------------
    if not all_embeddings:
        print("No embeddings were extracted! Exiting.")
        sys.exit(1)
        
    embeddings_array = np.array(all_embeddings)
    np.savez(
        EMBEDDINGS_FILE,
        embeddings=embeddings_array,
        labels=np.array(all_labels),
        paths=np.array(all_paths),
        splits=np.array(all_splits),
    )

    avg_ms = (total_time / total_images * 1000) if total_images else 0
    throughput = total_images / total_time if total_time else 0

    # Save speed metrics
    np.savez(
        SPEED_FILE,
        avg_inference_ms=avg_ms,
        throughput_ips=throughput,
        total_images=total_images,
        total_time_s=total_time,
    )

    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Mode:             {MODE}")
    print(f"  Total images:     {total_images}")
    print(f"  Embedding shape:  {embeddings_array.shape}")
    print(f"  Avg inference:    {avg_ms:.2f} ms/image")
    print(f"  Throughput:       {throughput:.1f} images/sec")
    if MODE == "raw":
        print(f"  Detection fails:  {fallback_count} (fell back to cropped)")
    print(f"  Saved embeddings: {EMBEDDINGS_FILE}")
    print(f"  Saved speed:      {SPEED_FILE}")


if __name__ == "__main__":
    main()
