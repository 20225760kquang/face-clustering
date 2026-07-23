"""
Leiden Algorithm Hyperparameter Grid Search Script.

Runs grid search over 'k' (KNN neighbors), 'min_sim' (cosine similarity threshold),
'resolution' (resolution parameter for CPM/RB partitions), and 'partition_type'
to find the optimal Leiden clustering parameters for face embeddings.

Usage:
    python benchmark/grid_search_leiden.py --model_name adaface --mode raw
    python benchmark/grid_search_leiden.py --model_name buffalo_l --mode raw --k_list 30 50 80 --min_sim_list 0.35 0.40 0.45 --resolution_list 0.005 0.01 0.05
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


def run_leiden(embeddings: np.ndarray, k: int = 50, min_sim: float = 0.40, resolution: float = 0.01, partition_type: str = "CPMVertexPartition", n_iterations: int = 2):
    """
    Runs Leiden community detection algorithm on KNN face graph.

    :param embeddings: Feature matrix (N, D)
    :param k: Top-K neighbors for KNN graph
    :param min_sim: Minimum cosine similarity threshold for graph edges
    :param resolution: Resolution parameter for CPM or RB partitions
    :param partition_type: Partition class ('CPMVertexPartition', 'ModularityVertexPartition', 'RBConfigurationVertexPartition')
    :param n_iterations: Number of Leiden refinement iterations (default 2, -1 for full convergence)
    :return: pred_labels (N,) numpy array
    """
    try:
        import igraph as ig
    except ImportError:
        raise ImportError("The 'python-igraph' package is required. Install via: pip install python-igraph igraph")

    try:
        import leidenalg
    except ImportError:
        leidenalg = None

    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    emb_norm = emb_norm.astype('float32')
    N, dim = emb_norm.shape

    # 1. Build KNN graph
    try:
        import faiss
        index = faiss.IndexFlatIP(dim)
        index.add(emb_norm)
        sims, nbrs = index.search(emb_norm, k=min(k, N))
    except ImportError:
        sim_matrix = emb_norm @ emb_norm.T
        nbrs = np.argsort(-sim_matrix, axis=1)[:, :min(k, N)]
        sims = np.take_along_axis(sim_matrix, nbrs, axis=1)

    # 2. Build edges with similarity threshold
    edges = []
    weights = []
    seen_edges = set()

    for i in range(N):
        for j_idx, sim in zip(nbrs[i], sims[i]):
            j = int(j_idx)
            if i == j:
                continue
            if sim >= min_sim:
                u, v = (i, j) if i < j else (j, i)
                if (u, v) not in seen_edges:
                    seen_edges.add((u, v))
                    edges.append((u, v))
                    weights.append(float(sim))

    # 3. Create igraph Graph
    g = ig.Graph(n=N, edges=edges, directed=False)
    if weights:
        g.es['weight'] = weights

    # 4. Run Leiden partitioning
    if leidenalg is not None:
        partition_cls = getattr(leidenalg, partition_type, leidenalg.CPMVertexPartition)
        if partition_type == "ModularityVertexPartition":
            partition = leidenalg.find_partition(
                g, partition_cls, weights='weight' if weights else None, n_iterations=n_iterations
            )
        else:
            partition = leidenalg.find_partition(
                g, partition_cls, weights='weight' if weights else None,
                resolution_parameter=resolution, n_iterations=n_iterations
            )
        membership = partition.membership
    else:
        # Fallback to igraph community_leiden
        part = g.community_leiden(weights='weight' if weights else None, resolution=resolution, n_iterations=n_iterations)
        membership = part.membership

    pred_labels = np.array(membership, dtype=int)
    return pred_labels


def main():
    parser = argparse.ArgumentParser(description="Leiden Hyperparameter Grid Search")
    parser.add_argument("--model_name", type=str, default="adaface", choices=["buffalo_l", "buffalo_m", "buffalo_sc", "finetuned", "adaface"])
    parser.add_argument("--mode", type=str, default="raw", choices=["raw", "cropped"])
    parser.add_argument("--k_list", type=int, nargs="+", default=[20, 30, 50, 80], help="List of K (KNN neighbors) values")
    parser.add_argument("--min_sim_list", type=float, nargs="+", default=[0.35, 0.38, 0.40, 0.42, 0.45, 0.50], help="List of min_sim threshold values")
    parser.add_argument("--resolution_list", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.05, 0.1], help="List of resolution parameters for Leiden")
    parser.add_argument("--partition_type", type=str, default="CPMVertexPartition", choices=["CPMVertexPartition", "ModularityVertexPartition", "RBConfigurationVertexPartition"], help="Leiden partition type")
    parser.add_argument("--output_dir", type=str, default="benchmark/results", help="Directory to save grid search report")
    args = parser.parse_args()

    embeddings, true_labels, paths, splits = load_embeddings(args.model_name, args.mode)
    n_persons = len(np.unique(true_labels))
    total_samples = len(embeddings)
    
    total_runs = len(args.k_list) * len(args.min_sim_list) * len(args.resolution_list)

    print("=" * 85)
    print(f" LEIDEN HYPERPARAMETER GRID SEARCH")
    print(f" Model: {args.model_name} | Mode: {args.mode}")
    print(f" Dataset: {total_samples} samples across {n_persons} true identities")
    print(f" Partition Type: {args.partition_type}")
    print(f" Grid parameters:")
    print(f"   - K (neighbors) : {args.k_list}")
    print(f"   - min_sim (th)   : {args.min_sim_list}")
    print(f"   - resolution     : {args.resolution_list}")
    print(f" Total combinations: {total_runs}")
    print("=" * 85)

    results = []
    run_idx = 0

    grid = itertools.product(args.k_list, args.min_sim_list, args.resolution_list)

    for k, min_sim, res in grid:
        run_idx += 1
        print(f"\n[{run_idx}/{total_runs}] Running Leiden (k={k}, min_sim={min_sim:.2f}, res={res}) ...", end="", flush=True)
        
        t0 = time.time()
        try:
            pred_labels = run_leiden(
                embeddings, k=k, min_sim=min_sim, resolution=res, partition_type=args.partition_type
            )
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
                "resolution": res,
                "partition_type": args.partition_type,
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
    print("\n" + "=" * 115)
    print(f" LEIDEN GRID SEARCH RESULTS (TOP RANKED BY BCUBED F1)")
    print("=" * 115)
    print(f"| Rank |   K   | min_sim | Resolution | Clusters | True IDs | BCubed F1 | BCubed P | BCubed R | Pairwise F1 |  ARI   |  NMI   | Time(s) |")
    print(f"|------|-------|---------|------------|----------|----------|-----------|----------|----------|-------------|--------|--------|---------|")

    for rank, r in enumerate(results, 1):
        print(
            f"| {rank:<4} | {r['k']:<5} | {r['min_sim']:<7.2f} | {r['resolution']:<10} | {r['n_clusters']:<8} | {n_persons:<8} | "
            f"{r['bcubed_f1']:<9.4f} | {r['bcubed_p']:<8.4f} | {r['bcubed_r']:<8.4f} | "
            f"{r['pairwise_f1']:<11.4f} | {r['ari']:<6.4f} | {r['nmi']:<6.4f} | {r['time_sec']:<7.2f} |"
        )
    print("=" * 115)

    if results:
        best = results[0]
        print(f"\nBEST PARAMETERS FOR {args.model_name.upper()} ({args.mode}):")
        print(f"  --> k = {best['k']}")
        print(f"  --> min_sim = {best['min_sim']}")
        print(f"  --> resolution = {best['resolution']}")
        print(f"  --> partition_type = {best['partition_type']}")
        print(f"  --> BCubed F1 = {best['bcubed_f1']:.4f}")
        print(f"  --> Clusters Found = {best['n_clusters']} (True IDs: {n_persons})")

    # Save all grid search runs into a single consolidated CSV file
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "leiden_grid_search_results.csv"

    fieldnames = [
        "timestamp", "model_name", "mode", "k", "min_sim", "resolution", "partition_type",
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
                "resolution": r["resolution"],
                "partition_type": r["partition_type"],
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
