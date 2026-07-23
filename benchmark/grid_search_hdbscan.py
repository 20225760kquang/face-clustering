"""
HDBSCAN Hyperparameter Grid Search Script.

Runs grid search over min_cluster_size, min_samples, cluster_selection_epsilon, and metric
to find the optimal HDBSCAN clustering parameters for face embeddings.

Usage:
    python benchmark/grid_search_hdbscan.py --model_name adaface --mode raw
    python benchmark/grid_search_hdbscan.py --model_name buffalo_l --mode raw --min_cluster_size_list 2 3 5 8 --min_samples_list 1 2 3
"""

import sys
import time
import argparse
import itertools
import numpy as np
from pathlib import Path
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Add benchmark directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from clustering import bcubed_metrics, pairwise_metrics


def load_embeddings(model_name: str, mode: str):
    emb_file = PROJECT_ROOT / "benchmark" / "embeddings" / f"{model_name}_{mode}_embeddings.npz"
    if not emb_file.exists():
        print(f"Error: Embeddings file not found at '{emb_file}'")
        print(f"Please run: python benchmark/extract_embeddings.py --model_name {model_name} --mode {mode}")
        sys.exit(1)
    
    print(f"Loading embeddings from: {emb_file.name}")
    data = np.load(emb_file, allow_pickle=True)
    return data["embeddings"], data["labels"], data["paths"], data["splits"]


def main():
    parser = argparse.ArgumentParser(description="HDBSCAN Hyperparameter Grid Search")
    parser.add_argument("--model_name", type=str, default="adaface", choices=["buffalo_l", "buffalo_m", "buffalo_sc", "finetuned", "adaface"])
    parser.add_argument("--mode", type=str, default="raw", choices=["raw", "cropped"])
    parser.add_argument("--min_cluster_size_list", type=int, nargs="+", default=[2, 3, 4, 5, 8, 10], help="List of min_cluster_size values")
    parser.add_argument("--min_samples_list", type=int, nargs="+", default=[1, 2, 3, 5], help="List of min_samples values")
    parser.add_argument("--epsilon_list", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25], help="List of cluster_selection_epsilon values")
    parser.add_argument("--metric_list", type=str, nargs="+", default=["euclidean"], help="List of distance metrics")
    parser.add_argument("--output_dir", type=str, default="benchmark/results", help="Directory to save grid search report")
    args = parser.parse_args()

    embeddings, true_labels, paths, splits = load_embeddings(args.model_name, args.mode)
    n_persons = len(np.unique(true_labels))
    total_samples = len(embeddings)
    
    total_runs = len(args.min_cluster_size_list) * len(args.min_samples_list) * len(args.epsilon_list) * len(args.metric_list)

    print("=" * 80)
    print(f" HDBSCAN HYPERPARAMETER GRID SEARCH")
    print(f" Model: {args.model_name} | Mode: {args.mode}")
    print(f" Dataset: {total_samples} samples across {n_persons} true identities")
    print(f" Grid parameters:")
    print(f"   - min_cluster_size        : {args.min_cluster_size_list}")
    print(f"   - min_samples             : {args.min_samples_list}")
    print(f"   - cluster_selection_eps   : {args.epsilon_list}")
    print(f"   - metric                  : {args.metric_list}")
    print(f" Total combinations: {total_runs}")
    print("=" * 80)

    results = []
    run_idx = 0

    grid = itertools.product(
        args.min_cluster_size_list,
        args.min_samples_list,
        args.epsilon_list,
        args.metric_list
    )

    for mcs, ms, eps, metric in grid:
        run_idx += 1
        param_str = f"mcs={mcs}, ms={ms}, eps={eps:.2f}, metric={metric}"
        print(f"\n[{run_idx}/{total_runs}] Running HDBSCAN ({param_str}) ...", end="", flush=True)
        
        t0 = time.time()
        try:
            hdb = HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_epsilon=eps,
                metric=metric
            )
            pred_labels = hdb.fit_predict(embeddings)
            t1 = time.time()
            elapsed = t1 - t0
            
            unique_preds = np.unique(pred_labels)
            n_clusters = int(np.sum(unique_preds >= 0))
            n_noise = int(np.sum(pred_labels == -1))
            noise_ratio = float(n_noise / len(pred_labels)) if len(pred_labels) > 0 else 0.0
            
            b_p, b_r, b_f1 = bcubed_metrics(true_labels, pred_labels)
            p_p, p_r, p_f1 = pairwise_metrics(true_labels, pred_labels)
            ari = float(adjusted_rand_score(true_labels, pred_labels))
            nmi = float(normalized_mutual_info_score(true_labels, pred_labels))
            
            print(f" Done ({elapsed:.2f}s) | Clusters: {n_clusters} | Noise: {n_noise} ({noise_ratio:.1%}) | BCubed F1: {b_f1:.4f}")
            
            results.append({
                "min_cluster_size": mcs,
                "min_samples": ms,
                "epsilon": eps,
                "metric": metric,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": noise_ratio,
                "bcubed_f1": b_f1,
                "bcubed_p": b_p,
                "bcubed_r": b_r,
                "pairwise_f1": p_f1,
                "pairwise_p": p_p,
                "pairwise_r": p_r,
                "ari": ari,
                "nmi": nmi,
                "time_sec": elapsed
            })
        except Exception as e:
            print(f" FAILED: {e}")

    # Sort results by BCubed F1 descending
    results = sorted(results, key=lambda x: x["bcubed_f1"], reverse=True)

    # Print Best Results Summary Table
    print("\n" + "=" * 115)
    print(f" HDBSCAN GRID SEARCH RESULTS (TOP RANKED BY BCUBED F1)")
    print("=" * 115)
    print(f"| Rank | mcs | ms |  eps  |  metric   | Clusters | Noise% | BCubed F1 | BCubed P | BCubed R | Pairwise F1 |  ARI   |  NMI   | Time(s) |")
    print(f"|------|-----|----|-------|-----------|----------|--------|-----------|----------|----------|-------------|--------|--------|---------|")

    for rank, r in enumerate(results, 1):
        noise_pct = f"{r['noise_ratio']*100:.1f}%"
        print(
            f"| {rank:<4} | {r['min_cluster_size']:<3} | {r['min_samples']:<2} | {r['epsilon']:<5.2f} | {r['metric']:<9} | "
            f"{r['n_clusters']:<8} | {noise_pct:<6} | {r['bcubed_f1']:<9.4f} | {r['bcubed_p']:<8.4f} | {r['bcubed_r']:<8.4f} | "
            f"{r['pairwise_f1']:<11.4f} | {r['ari']:<6.4f} | {r['nmi']:<6.4f} | {r['time_sec']:<7.2f} |"
        )
    print("=" * 115)

    if results:
        best = results[0]
        print(f"\nBEST PARAMETERS FOR {args.model_name.upper()} ({args.mode}):")
        print(f"  --> min_cluster_size = {best['min_cluster_size']}")
        print(f"  --> min_samples = {best['min_samples']}")
        print(f"  --> cluster_selection_epsilon = {best['epsilon']}")
        print(f"  --> metric = {best['metric']}")
        print(f"  --> BCubed F1 = {best['bcubed_f1']:.4f}")
        print(f"  --> Clusters Found = {best['n_clusters']} (True IDs: {n_persons})")

    # Save all grid search runs into a single consolidated CSV file
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "hdbscan_grid_search_results.csv"

    fieldnames = [
        "timestamp", "model_name", "mode", "min_cluster_size", "min_samples", "epsilon", "metric",
        "n_clusters", "n_noise", "noise_ratio", "true_identities", "bcubed_f1", "bcubed_precision", "bcubed_recall",
        "pairwise_f1", "pairwise_precision", "pairwise_recall", "ari", "nmi", "time_sec"
    ]

    import csv
    from datetime import datetime

    file_exists = csv_file.exists()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_file, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for r in results:
            writer.writerow({
                "timestamp": timestamp_str,
                "model_name": args.model_name,
                "mode": args.mode,
                "min_cluster_size": r["min_cluster_size"],
                "min_samples": r["min_samples"],
                "epsilon": r["epsilon"],
                "metric": r["metric"],
                "n_clusters": r["n_clusters"],
                "n_noise": r["n_noise"],
                "noise_ratio": round(r["noise_ratio"], 4),
                "true_identities": n_persons,
                "bcubed_f1": round(r["bcubed_f1"], 4),
                "bcubed_precision": round(r["bcubed_p"], 4),
                "bcubed_recall": round(r["bcubed_r"], 4),
                "pairwise_f1": round(r["pairwise_f1"], 4),
                "pairwise_precision": round(r["pairwise_p"], 4),
                "pairwise_recall": round(r["pairwise_r"], 4),
                "ari": round(r["ari"], 4),
                "nmi": round(r["nmi"], 4),
                "time_sec": round(r["time_sec"], 2)
            })

    print(f"\nAll HDBSCAN experiments appended & saved to CSV: {csv_file}")


if __name__ == "__main__":
    main()
