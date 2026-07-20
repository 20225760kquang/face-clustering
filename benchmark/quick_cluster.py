"""
Quick Cluster Evaluator for HDBSCAN and K-Means.

Designed for fast testing on extracted embeddings without running a lengthy multi-algorithm grid search.

Usage:
    python benchmark/quick_cluster.py --model_name buffalo_l --mode raw
    python benchmark/quick_cluster.py --model_name finetuned --mode raw --algorithms kmeans hdbscan
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.cluster import HDBSCAN, KMeans, MiniBatchKMeans

# Add parent & benchmark directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import cfg
from clustering import bcubed_metrics, pairwise_metrics
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def load_embeddings(model_name: str, mode: str):
    emb_file = PROJECT_ROOT / "benchmark" / "embeddings" / f"{model_name}_{mode}_embeddings.npz"
    if not emb_file.exists():
        print(f"Error: Embeddings file not found at '{emb_file}'")
        print(f"Please run: python benchmark/extract_embeddings.py --model_name {model_name} --mode {mode}")
        sys.exit(1)
    
    print(f"Loading embeddings from: {emb_file.name}")
    data = np.load(emb_file, allow_pickle=True)
    return data["embeddings"], data["labels"], data["paths"], data["splits"]


def evaluate_cluster_labels(algo_name, param_str, true_labels, pred_labels, elapsed_sec):
    unique_preds = np.unique(pred_labels)
    n_clusters = int(np.sum(unique_preds >= 0))
    n_noise = int(np.sum(pred_labels == -1))
    noise_ratio = float(n_noise / len(pred_labels)) if len(pred_labels) > 0 else 0.0

    b_p, b_r, b_f1 = bcubed_metrics(true_labels, pred_labels)
    p_p, p_r, p_f1 = pairwise_metrics(true_labels, pred_labels)
    ari = float(adjusted_rand_score(true_labels, pred_labels))
    nmi = float(normalized_mutual_info_score(true_labels, pred_labels))

    print(f"\n[{algo_name}] ({param_str}) - Completed in {elapsed_sec:.2f}s")
    print(f"  Clusters Found:    {n_clusters} (True identities: {len(np.unique(true_labels))})")
    print(f"  Noise Count:       {n_noise} ({noise_ratio:.1%})")
    print(f"  BCubed Metrics:    F1={b_f1:.4f} | Precision={b_p:.4f} | Recall={b_r:.4f}")
    print(f"  Pairwise Metrics:  F1={p_f1:.4f} | Precision={p_p:.4f} | Recall={p_r:.4f}")
    print(f"  ARI: {ari:.4f} | NMI: {nmi:.4f}")

    return {
        "algorithm": algo_name,
        "param_str": param_str,
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "bcubed_f1": b_f1,
        "bcubed_p": b_p,
        "bcubed_r": b_r,
        "pairwise_f1": p_f1,
        "pairwise_p": p_p,
        "pairwise_r": p_r,
        "ari": ari,
        "nmi": nmi,
        "time_sec": elapsed_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick HDBSCAN & K-Means Cluster Evaluator")
    parser.add_argument("--model_name", type=str, default="buffalo_l", choices=["buffalo_l", "buffalo_m", "buffalo_sc", "finetuned"])
    parser.add_argument("--mode", type=str, default="raw", choices=["raw", "cropped"])
    parser.add_argument("--algorithms", nargs="+", default=["kmeans", "hdbscan"], choices=["kmeans", "hdbscan"])
    parser.add_argument("--min_cluster_size", type=int, default=3, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min_samples", type=int, default=1, help="HDBSCAN min_samples")
    parser.add_argument("--n_clusters", type=int, default=None, help="KMeans n_clusters (default: auto set to true identity count)")
    parser.add_argument("--minibatch", action="store_true", default=True, help="Use MiniBatchKMeans for 10x-50x faster speed")
    parser.add_argument("--full_kmeans", action="store_true", help="Force standard full-dataset KMeans (slower)")
    args = parser.parse_args()

    embeddings, true_labels, paths, splits = load_embeddings(args.model_name, args.mode)
    n_persons = len(np.unique(true_labels))
    print(f"Dataset summary: {len(embeddings)} images across {n_persons} unique persons.")

    results = []

    # 1. K-Means Evaluation
    if "kmeans" in args.algorithms:
        k = args.n_clusters if args.n_clusters is not None else n_persons
        use_minibatch = not args.full_kmeans

        if use_minibatch:
            print(f"\n--> Running Fast MiniBatchKMeans (k={k}, batch_size=2048) ...")
            t0 = time.time()
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048, random_state=42, n_init="auto")
            pred_labels = kmeans.fit_predict(embeddings)
            t1 = time.time()
            res = evaluate_cluster_labels("MiniBatchKMeans", f"n_clusters={k}, batch_size=2048", true_labels, pred_labels, t1 - t0)
        else:
            print(f"\n--> Running Standard KMeans (k={k}) ...")
            t0 = time.time()
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            pred_labels = kmeans.fit_predict(embeddings)
            t1 = time.time()
            res = evaluate_cluster_labels("KMeans", f"n_clusters={k}", true_labels, pred_labels, t1 - t0)
        
        results.append(res)

    # 2. HDBSCAN Evaluation
    if "hdbscan" in args.algorithms:
        mcs = args.min_cluster_size
        ms = args.min_samples
        print(f"\n--> Running HDBSCAN (min_cluster_size={mcs}, min_samples={ms}) ...")
        t0 = time.time()
        hdb = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean")
        pred_labels = hdb.fit_predict(embeddings)
        t1 = time.time()
        res = evaluate_cluster_labels("HDBSCAN", f"min_cluster_size={mcs}, min_samples={ms}", true_labels, pred_labels, t1 - t0)
        results.append(res)

    print("\n" + "=" * 60)
    print("QUICK CLUSTERING EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
