"""
Quick Cluster Evaluator for HDBSCAN and K-Means.

Designed for fast testing on extracted embeddings without running a lengthy multi-algorithm grid search.

Usage:
    python benchmark/quick_cluster.py --model_name buffalo_l --mode raw (default = MiniBatchKMeans)
    python benchmark/quick_cluster.py --model_name buffalo_l --mode raw --full_kmeans (force standard KMeans)
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

import csv
from clustering import bcubed_metrics, pairwise_metrics, run_infomap, post_process_merge_clusters
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


def export_clusters_to_csv(model_name: str, mode: str, algo_name: str, paths: np.ndarray, true_labels: np.ndarray, pred_labels: np.ndarray, param_suffix: str = "", output_dir: str = "benchmark/results"):
    """
    Exports cluster assignments (Clusters 1 to N) to a CSV file with detailed parameter tags in filename.
    """
    unique_clusters = np.unique(pred_labels)
    non_noise_clusters = sorted([int(c) for c in unique_clusters if c >= 0])
    if -1 in unique_clusters:
        non_noise_clusters.append(-1)
        
    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if param_suffix:
        csv_file = out_dir / f"{model_name}_{mode}_{algo_name.lower()}_clusters_detail_{param_suffix}.csv"
    else:
        csv_file = out_dir / f"{model_name}_{mode}_{algo_name.lower()}_clusters_detail.csv"
    
    fieldnames = [
        "cluster_num",
        "cluster_id",
        "total_images",
        "dominant_gt_id",
        "purity",
        "gt_id_breakdown"
    ]
    
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, c_id in enumerate(non_noise_clusters, 1):
            mask = (pred_labels == c_id)
            c_labels = true_labels[mask]
            total_imgs = len(c_labels)
            
            unique_gt, counts = np.unique(c_labels, return_counts=True)
            gt_counts = sorted(zip(unique_gt, counts), key=lambda x: x[1], reverse=True)
            
            dominant_gt_id = str(gt_counts[0][0]) if len(gt_counts) > 0 else "N/A"
            dominant_count = gt_counts[0][1] if len(gt_counts) > 0 else 0
            purity = round(dominant_count / total_imgs, 4) if total_imgs > 0 else 0.0
            
            breakdown_str = "; ".join([f"{gt}: {cnt}" for gt, cnt in gt_counts])
            
            writer.writerow({
                "cluster_num": idx if c_id >= 0 else "NOISE",
                "cluster_id": c_id,
                "total_images": total_imgs,
                "dominant_gt_id": dominant_gt_id,
                "purity": purity,
                "gt_id_breakdown": breakdown_str
            })
            
    n_clusters_count = len([c for c in non_noise_clusters if c >= 0])
    print(f"\n[ALERT] Detailed cluster output (Clusters 1 to {n_clusters_count}) saved to CSV:")
    print(f"        --> {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Quick Cluster Evaluator (K-Means, HDBSCAN, Infomap)")
    parser.add_argument("--model_name", type=str, default="buffalo_l", choices=["buffalo_l", "buffalo_m", "buffalo_sc", "finetuned", "adaface"])
    parser.add_argument("--mode", type=str, default="raw", choices=["raw", "cropped"])
    parser.add_argument("--algorithms", nargs="+", default=["kmeans", "hdbscan", "infomap"], choices=["kmeans", "hdbscan", "infomap"])
    parser.add_argument("--min_cluster_size", type=int, default=3, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min_samples", type=int, default=1, help="HDBSCAN min_samples")
    parser.add_argument("--infomap_k", type=int, default=50, help="Infomap top-K neighbors")
    parser.add_argument("--infomap_min_sim", type=float, default=0.58, help="Infomap min similarity threshold")
    parser.add_argument("--n_clusters", type=int, default=None, help="KMeans n_clusters (default: auto set to true identity count)")
    parser.add_argument("--minibatch", action="store_true", default=False, help="Use MiniBatchKMeans instead of standard KMeans for faster speed")
    parser.add_argument("--post_process", action="store_true", default=False, help="Enable post-processing centroid merging to merge fragmented clusters")
    parser.add_argument("--merge_sim", type=float, default=0.55, help="Cosine similarity threshold for centroid merging in post-processing")
    args = parser.parse_args()

    embeddings, true_labels, paths, splits = load_embeddings(args.model_name, args.mode)
    n_persons = len(np.unique(true_labels))
    print(f"Dataset summary: {len(embeddings)} images across {n_persons} unique persons.")

    results = []

    # 1. K-Means Evaluation
    if "kmeans" in args.algorithms:
        k = args.n_clusters if args.n_clusters is not None else n_persons
        use_minibatch = args.minibatch

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

        if args.post_process:
            pred_labels = post_process_merge_clusters(embeddings, pred_labels, merge_sim=args.merge_sim)
            res = evaluate_cluster_labels("KMeans [Post-Processed]", f"n_clusters={k}, merge_sim={args.merge_sim}", true_labels, pred_labels, 0.0)

        algo_title = "MiniBatchKMeans" if use_minibatch else "KMeans"
        param_tag = f"k_{k}" if not args.post_process else f"k_{k}_merged_{args.merge_sim}"
        export_clusters_to_csv(args.model_name, args.mode, algo_title, paths, true_labels, pred_labels, param_suffix=param_tag)
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
        
        if args.post_process:
            pred_labels = post_process_merge_clusters(embeddings, pred_labels, merge_sim=args.merge_sim)
            res = evaluate_cluster_labels("HDBSCAN [Post-Processed]", f"mcs={mcs}, ms={ms}, merge_sim={args.merge_sim}", true_labels, pred_labels, 0.0)

        param_tag = f"mcs_{mcs}_ms_{ms}" if not args.post_process else f"mcs_{mcs}_ms_{ms}_merged_{args.merge_sim}"
        export_clusters_to_csv(args.model_name, args.mode, "HDBSCAN", paths, true_labels, pred_labels, param_suffix=param_tag)
        results.append(res)

    # 3. Infomap Evaluation
    if "infomap" in args.algorithms:
        ik = args.infomap_k
        min_sim = args.infomap_min_sim
        print(f"\n--> Running Infomap (k={ik}, min_sim={min_sim}) ...")
        t0 = time.time()
        try:
            pred_labels = run_infomap(embeddings, k=ik, min_sim=min_sim)
            t1 = time.time()
            res = evaluate_cluster_labels("Infomap", f"k={ik}, min_sim={min_sim}", true_labels, pred_labels, t1 - t0)
            
            min_sim_str = f"{min_sim:.2f}".rstrip('0').rstrip('.') if isinstance(min_sim, float) and len(str(min_sim).split('.')[-1]) > 2 else f"{min_sim}"
            param_tag = f"k_{ik}_min_sim_{min_sim_str}"

            if args.post_process:
                print(f"\n--> Applying Post-Processing Centroid Merging (merge_sim={args.merge_sim}) ...")
                pred_labels = post_process_merge_clusters(embeddings, pred_labels, merge_sim=args.merge_sim)
                res = evaluate_cluster_labels("Infomap [Post-Processed]", f"k={ik}, min_sim={min_sim}, merge_sim={args.merge_sim}", true_labels, pred_labels, 0.0)
                param_tag += f"_merged_{args.merge_sim}"

            export_clusters_to_csv(args.model_name, args.mode, "Infomap", paths, true_labels, pred_labels, param_suffix=param_tag)
            results.append(res)
        except Exception as e:
            print(f"  Failed to run Infomap: {e}")


    print("\n" + "=" * 60)
    print("QUICK CLUSTERING EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
