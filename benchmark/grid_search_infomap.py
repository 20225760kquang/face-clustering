"""
Infomap Hyperparameter Grid Search Script.

Runs grid search over various values of 'k' (KNN neighbors) and 'min_sim' (cosine similarity threshold)
to find the optimal Infomap clustering parameters for face embeddings.

Usage:
    python benchmark/grid_search_infomap.py --model_name adaface --mode raw
    python benchmark/grid_search_infomap.py --model_name buffalo_l --mode raw --k_list 30 50 80 --min_sim_list 0.35 0.38 0.40 0.42 0.45
"""

import sys
import time
import argparse
import itertools
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Add benchmark directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from clustering import bcubed_metrics, pairwise_metrics, run_infomap


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
    parser = argparse.ArgumentParser(description="Infomap Hyperparameter Grid Search")
    parser.add_argument("--model_name", type=str, default="adaface", choices=["buffalo_l", "buffalo_m", "buffalo_sc", "finetuned", "adaface"])
    parser.add_argument("--mode", type=str, default="raw", choices=["raw", "cropped"])
    parser.add_argument("--k_list", type=int, nargs="+", default=[20, 30, 50, 80, 100], help="List of K (KNN neighbors) values")
    parser.add_argument("--min_sim_list", type=float, nargs="+", default=[0.32, 0.35, 0.38, 0.40, 0.42, 0.45, 0.50, 0.55], help="List of min_sim threshold values")
    parser.add_argument("--output_dir", type=str, default="benchmark/results", help="Directory to save grid search report")
    args = parser.parse_args()

    embeddings, true_labels, paths, splits = load_embeddings(args.model_name, args.mode)
    n_persons = len(np.unique(true_labels))
    total_samples = len(embeddings)
    
    print("=" * 75)
    print(f" INFOMAP HYPERPARAMETER GRID SEARCH")
    print(f" Model: {args.model_name} | Mode: {args.mode}")
    print(f" Dataset: {total_samples} samples across {n_persons} true identities")
    print(f" Grid parameters:")
    print(f"   - K (neighbors) : {args.k_list}")
    print(f"   - min_sim (th)   : {args.min_sim_list}")
    print(f" Total combinations: {len(args.k_list) * len(args.min_sim_list)}")
    print("=" * 75)

    results = []
    total_runs = len(args.k_list) * len(args.min_sim_list)
    run_idx = 0

    for k, min_sim in itertools.product(args.k_list, args.min_sim_list):
        run_idx += 1
        print(f"\n[{run_idx}/{total_runs}] Running Infomap (k={k}, min_sim={min_sim:.2f}) ...", end="", flush=True)
        
        t0 = time.time()
        try:
            pred_labels = run_infomap(embeddings, k=k, min_sim=min_sim)
            t1 = time.time()
            elapsed = t1 - t0
            
            unique_preds = np.unique(pred_labels)
            n_clusters = int(np.sum(unique_preds >= 0))
            
            b_p, b_r, b_f1 = bcubed_metrics(true_labels, pred_labels)
            p_p, p_r, p_f1 = pairwise_metrics(true_labels, pred_labels)
            ari = float(adjusted_rand_score(true_labels, pred_labels))
            nmi = float(normalized_mutual_info_score(true_labels, pred_labels))
            
            print(f" Done ({elapsed:.2f}s) | Clusters: {n_clusters} | BCubed F1: {b_f1:.4f} | Pairwise F1: {p_f1:.4f}")
            
            results.append({
                "k": k,
                "min_sim": min_sim,
                "n_clusters": n_clusters,
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
    print("\n" + "=" * 105)
    print(f" INFOMAP GRID SEARCH RESULTS (TOP RANGED BY BCUBED F1)")
    print("=" * 105)
    print(f"| Rank |   K   | min_sim | Clusters | True IDs | BCubed F1 | BCubed P | BCubed R | Pairwise F1 |  ARI   |  NMI   | Time(s) |")
    print(f"|------|-------|---------|----------|----------|-----------|----------|----------|-------------|--------|--------|---------|")

    for rank, r in enumerate(results, 1):
        print(
            f"| {rank:<4} | {r['k']:<5} | {r['min_sim']:<7.2f} | {r['n_clusters']:<8} | {n_persons:<8} | "
            f"{r['bcubed_f1']:<9.4f} | {r['bcubed_p']:<8.4f} | {r['bcubed_r']:<8.4f} | "
            f"{r['pairwise_f1']:<11.4f} | {r['ari']:<6.4f} | {r['nmi']:<6.4f} | {r['time_sec']:<7.2f} |"
        )
    print("=" * 105)

    if results:
        best = results[0]
        print(f"\nBEST PARAMETERS FOR {args.model_name.upper()} ({args.mode}):")
        print(f"  --> k = {best['k']}")
        print(f"  --> min_sim = {best['min_sim']}")
        print(f"  --> BCubed F1 = {best['bcubed_f1']:.4f}")
        print(f"  --> Clusters Found = {best['n_clusters']} (True IDs: {n_persons})")

    # Save all grid search runs into a single consolidated CSV file
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "infomap_grid_search_results.csv"

    fieldnames = [
        "timestamp", "model_name", "mode", "k", "min_sim",
        "n_clusters", "true_identities", "bcubed_f1", "bcubed_precision", "bcubed_recall",
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
                "k": r["k"],
                "min_sim": r["min_sim"],
                "n_clusters": r["n_clusters"],
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

    print(f"\nAll experiments appended & saved to CSV: {csv_file}")


if __name__ == "__main__":
    main()

