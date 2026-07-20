"""
Phase 2: Evaluate face embeddings — Verification, Identification, Clustering.

Loads pre-computed embeddings from .npz (Phase 1) and runs:
  1. Face Verification (1:1)  — ROC-AUC, TAR@FAR, EER
  2. Face Identification (1:N) — Rank-1/5/10, CMC, mAP
  3. Clustering Quality        — Silhouette, intra/inter distance, t-SNE, UMAP, and grid search clustering metrics.

Usage:
    python evaluate.py
    python evaluate.py --mode cropped
    python evaluate.py --model_name finetuned
"""

import json
import sys
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, silhouette_score

# Import configurations from config.py
from config import cfg

# Import clustering tools
from clustering import ClusteringEvaluator, generate_markdown_table

# ---------------------------------------------------------------------------
# Paths and Seeds
# ---------------------------------------------------------------------------
EMBEDDINGS_PATH = cfg["extraction"]["embeddings_file"]
SPEED_PATH = cfg["extraction"]["speed_file"]
RESULTS_DIR = cfg["clustering"]["output_path"]

MODEL_NAME = cfg["extraction"]["model_name"]
MODE = cfg["extraction"]["mode"]

REPORT_MD = cfg["clustering"]["report_md"]
REPORT_JSON = cfg["clustering"]["report_json"]

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# General Helpers
# ---------------------------------------------------------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between rows of A (M,D) and B (N,D) → (M,N)."""
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_n @ B_n.T


def load_data():
    if not EMBEDDINGS_PATH.exists():
        print(f"Error: Embeddings file not found at '{EMBEDDINGS_PATH}'")
        print("Please run extract_embeddings.py first to generate embeddings.")
        sys.exit(1)
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return data["embeddings"], data["labels"], data["paths"], data["splits"]


# =========================================================================
# Class-based Evaluators
# =========================================================================

class FaceVerificationEval:
    """
    Evaluates Face Verification (1:1) performance.
    Calculates ROC-AUC, TAR@FAR, EER, best threshold, and best accuracy.
    Generates ROC & Similarity Distribution plots.
    """
    def __init__(self, embeddings, labels, results_dir, model_name, mode):
        self.embeddings = embeddings
        self.labels = labels
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        self.mode = mode

    def evaluate(self):
        print("\n" + "=" * 60)
        print("1. FACE VERIFICATION (1:1)")
        print("=" * 60)

        # Group indices by person
        person_idx = defaultdict(list)
        for i, lbl in enumerate(self.labels):
            person_idx[lbl].append(i)
        persons = list(person_idx.keys())

        # ---- Positive pairs (same person) ----
        pos_pairs = []
        for indices in person_idx.values():
            if len(indices) >= 2:
                pos_pairs.extend(combinations(indices, 2))
        print(f"  Positive pairs: {len(pos_pairs)}")

        # ---- Negative pairs (different person), balanced ----
        neg_pairs = []
        target = len(pos_pairs)
        while len(neg_pairs) < target:
            p1, p2 = random.sample(persons, 2)
            i = random.choice(person_idx[p1])
            j = random.choice(person_idx[p2])
            neg_pairs.append((i, j))
        print(f"  Negative pairs: {len(neg_pairs)}")

        # ---- Compute cosine similarities ----
        pos_sims = np.array([cosine_sim(self.embeddings[i], self.embeddings[j]) for i, j in pos_pairs])
        neg_sims = np.array([cosine_sim(self.embeddings[i], self.embeddings[j]) for i, j in neg_pairs])

        all_sims = np.concatenate([pos_sims, neg_sims])
        all_y = np.array([1] * len(pos_sims) + [0] * len(neg_sims))

        # ---- ROC ----
        fpr, tpr, thresholds = roc_curve(all_y, all_sims)
        roc_auc = auc(fpr, tpr)

        # TAR @ FAR
        tar_at_001 = float(tpr[np.searchsorted(fpr, 0.01)])
        tar_at_0001 = float(tpr[np.searchsorted(fpr, 0.001)])

        # EER
        fnr = 1 - tpr
        eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float(fpr[eer_idx])

        # Best threshold (Youden's J)
        best_idx = int(np.argmax(tpr - fpr))
        best_thr = float(thresholds[best_idx])
        preds = np.concatenate([
            (pos_sims >= best_thr).astype(int),
            (neg_sims < best_thr).astype(int),
        ])
        best_acc = float(preds.mean())

        results = {
            "roc_auc": float(roc_auc),
            "tar_at_far_0.01": tar_at_001,
            "tar_at_far_0.001": tar_at_0001,
            "eer": eer,
            "best_threshold": best_thr,
            "best_accuracy": best_acc,
            "n_positive_pairs": len(pos_pairs),
            "n_negative_pairs": len(neg_pairs),
        }

        print(f"  ROC-AUC:          {roc_auc:.4f}")
        print(f"  TAR@FAR=0.01:     {tar_at_001:.4f}")
        print(f"  TAR@FAR=0.001:    {tar_at_0001:.4f}")
        print(f"  EER:              {eer:.4f}")
        print(f"  Best threshold:   {best_thr:.4f}")
        print(f"  Best accuracy:    {best_acc:.4f}")

        # ---- Plots ----
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC curve
        axes[0].plot(fpr, tpr, "b-", lw=2, label=f"{self.model_name} ({self.mode}) (AUC={roc_auc:.4f})")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve — Face Verification")
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)

        # Similarity distribution
        axes[1].hist(pos_sims, bins=60, alpha=0.7, label="Same Person", color="green", density=True)
        axes[1].hist(neg_sims, bins=60, alpha=0.7, label="Different Person", color="red", density=True)
        axes[1].axvline(best_thr, color="blue", ls="--", label=f"Threshold={best_thr:.3f}")
        axes[1].set_xlabel("Cosine Similarity")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Similarity Distribution")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.results_dir / f"{self.model_name}_{self.mode}_verification.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {plot_path.name}")

        return results


class FaceIdentificationEval:
    """
    Evaluates Face Identification (1:N) performance.
    Calculates Rank-1, Rank-5, Rank-10 accuracy, mAP, and plots the CMC Curve.
    """
    def __init__(self, embeddings, labels, splits, results_dir, model_name, mode):
        self.embeddings = embeddings
        self.labels = labels
        self.splits = splits
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        self.mode = mode

    def evaluate(self):
        print("\n" + "=" * 60)
        print("2. FACE IDENTIFICATION (1:N)")
        print("=" * 60)

        gallery_mask = self.splits == "gallery"
        query_mask = self.splits == "query"

        gallery_embs = self.embeddings[gallery_mask]
        gallery_labels = self.labels[gallery_mask]
        query_embs = self.embeddings[query_mask]
        query_labels = self.labels[query_mask]

        print(f"  Gallery: {len(gallery_embs)} images")
        print(f"  Query:   {len(query_embs)} images")

        if len(gallery_embs) == 0 or len(query_embs) == 0:
            print("  Warning: Empty gallery or query set. Skipping 1:N evaluation.")
            return {
                "rank_1_accuracy": 0.0,
                "rank_5_accuracy": 0.0,
                "rank_10_accuracy": 0.0,
                "mAP": 0.0,
                "n_gallery": 0,
                "n_query": 0,
                "n_persons": 0
            }

        # ---- Build gallery centroids ----
        unique_persons = np.unique(gallery_labels)
        centroid_list, centroid_ids = [], []
        for person in unique_persons:
            embs = gallery_embs[gallery_labels == person]
            centroid = embs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            centroid_list.append(centroid)
            centroid_ids.append(person)
        centroid_matrix = np.array(centroid_list)

        # ---- Similarity: each query vs all centroids ----
        sim_matrix = cosine_sim_matrix(query_embs, centroid_matrix)  # (n_query, n_persons)

        max_rank = min(20, len(centroid_ids))
        correct_at_rank = np.zeros(max_rank)
        avg_precisions = []

        for q in range(len(query_embs)):
            true_person = query_labels[q]
            ranked = np.argsort(sim_matrix[q])[::-1]
            ranked_persons = [centroid_ids[r] for r in ranked]

            for rank in range(max_rank):
                if true_person in ranked_persons[: rank + 1]:
                    correct_at_rank[rank] += 1

            if true_person in ranked_persons:
                ap = 1.0 / (ranked_persons.index(true_person) + 1)
            else:
                ap = 0.0
            avg_precisions.append(ap)

        n_q = len(query_embs)
        cmc = correct_at_rank / n_q
        mAP = float(np.mean(avg_precisions))

        results = {
            "rank_1_accuracy": float(cmc[0]),
            "rank_5_accuracy": float(cmc[4]) if max_rank > 4 else None,
            "rank_10_accuracy": float(cmc[9]) if max_rank > 9 else None,
            "mAP": mAP,
            "n_gallery": int(len(gallery_embs)),
            "n_query": int(n_q),
            "n_persons": int(len(unique_persons)),
        }

        print(f"  Rank-1 Accuracy:  {cmc[0]:.4f}")
        if max_rank > 4:
            print(f"  Rank-5 Accuracy:  {cmc[4]:.4f}")
        if max_rank > 9:
            print(f"  Rank-10 Accuracy: {cmc[9]:.4f}")
        print(f"  mAP:              {mAP:.4f}")

        # ---- CMC Curve ----
        fig, ax = plt.subplots(figsize=(8, 5))
        ranks = np.arange(1, max_rank + 1)
        ax.plot(ranks, cmc, "b-o", ms=4, lw=2)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Identification Rate")
        ax.set_title("CMC Curve — Face Identification")
        ax.set_xticks(ranks if max_rank <= 20 else ranks[::2])
        ax.set_ylim([max(0, cmc[0] - 0.15), 1.02])
        ax.grid(True, alpha=0.3)
        ax.annotate(
            f"Rank-1: {cmc[0]:.2%}",
            xy=(1, cmc[0]),
            xytext=(3, cmc[0] - 0.08),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10,
            color="red",
        )
        plt.tight_layout()
        plot_path = self.results_dir / f"{self.model_name}_{self.mode}_cmc.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {plot_path.name}")

        return results


class FaceClusteringEval:
    """
    Evaluates Embedding Space Quality and runs a Grid Search Clustering Benchmark.
    Generates Silhouette scores, Intra/Inter-class distances, t-SNE and UMAP visualizations,
    and runs clustering algorithm grid searches (reporting BCubed and Pairwise F1 metrics).
    """
    def __init__(self, embeddings, labels, results_dir, model_name, mode):
        self.embeddings = embeddings
        self.labels = labels
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        self.mode = mode

    def evaluate_space_quality(self, tsne_perplexity=30):
        print("\n" + "=" * 60)
        print("3. EMBEDDING SPACE QUALITY & VISUALIZATION")
        print("=" * 60)

        unique_labels = np.unique(self.labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        numeric = np.array([label_map[l] for l in self.labels])

        # Silhouette (cosine)
        sil = silhouette_score(self.embeddings, numeric, metric="cosine")

        # Intra / Inter class cosine distance (distance = 1 - similarity)
        person_idx = defaultdict(list)
        for i, lbl in enumerate(self.labels):
            person_idx[lbl].append(i)
        persons = list(person_idx.keys())

        intra_dists = []
        for indices in person_idx.values():
            if len(indices) < 2:
                continue
            for i, j in combinations(indices, 2):
                intra_dists.append(1.0 - cosine_sim(self.embeddings[i], self.embeddings[j]))

        inter_dists = []
        n_samples = min(len(intra_dists) * 2, 50_000)
        for _ in range(n_samples):
            p1, p2 = random.sample(persons, 2)
            i = random.choice(person_idx[p1])
            j = random.choice(person_idx[p2])
            inter_dists.append(1.0 - cosine_sim(self.embeddings[i], self.embeddings[j]))

        mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
        mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
        ratio = mean_inter / mean_intra if mean_intra > 0 else float("inf")

        results = {
            "silhouette_score": float(sil),
            "mean_intra_class_distance": mean_intra,
            "mean_inter_class_distance": mean_inter,
            "inter_intra_ratio": ratio,
        }

        print(f"  Silhouette Score:   {sil:.4f}")
        print(f"  Intra-class dist:   {mean_intra:.4f}  (lower = better)")
        print(f"  Inter-class dist:   {mean_inter:.4f}  (higher = better)")
        print(f"  Inter/Intra ratio:  {ratio:.2f}x")

        # ---- t-SNE visualization ----
        print("  Running t-SNE ...")
        # Adjust perplexity if dataset is too small
        if len(self.embeddings) <= tsne_perplexity:
            tsne_perplexity = max(2, len(self.embeddings) // 2)

        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, metric="cosine")
        tsne_2d = tsne.fit_transform(self.embeddings)

        self._save_scatter_plot(tsne_2d, unique_labels, "t-SNE", f"{self.model_name}_{self.mode}_tsne.png")

        # ---- UMAP visualization ----
        print("  Running UMAP ...")
        try:
            import umap
            n_neighbors = 15
            if len(self.embeddings) <= n_neighbors:
                n_neighbors = max(2, len(self.embeddings) - 1)
            
            reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, metric="cosine", random_state=42)
            umap_2d = reducer.fit_transform(self.embeddings)
            
            self._save_scatter_plot(umap_2d, unique_labels, "UMAP", f"{self.model_name}_{self.mode}_umap.png")
        except Exception as e:
            print(f"  Warning: UMAP generation failed or 'umap-learn' not installed properly: {e}")

        return results

    def _save_scatter_plot(self, coords_2d, unique_labels, method_name, filename):
        n_colors = min(20, len(unique_labels))
        viz_labels = unique_labels[:n_colors]
        cmap = plt.colormaps.get_cmap("tab20").resampled(n_colors)

        fig, ax = plt.subplots(figsize=(12, 10))
        for idx, person in enumerate(viz_labels):
            mask = self.labels == person
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[cmap(idx)], label=person, s=20, alpha=0.7)

        rest_mask = ~np.isin(self.labels, viz_labels)
        if rest_mask.any():
            ax.scatter(
                coords_2d[rest_mask, 0], coords_2d[rest_mask, 1],
                c="lightgray", s=10, alpha=0.3,
                label=f"Others ({rest_mask.sum()} pts)",
            )

        ax.set_title(f"{method_name} — Face Embeddings ({len(unique_labels)} persons)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plot_path = self.results_dir / filename
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {plot_path.name}")

    def run_grid_search(self, clustering_config):
        print("\n" + "=" * 60)
        print("4. RUNNING CLUSTERING GRID SEARCH BENCHMARK")
        print("=" * 60)
        evaluator = ClusteringEvaluator(self.embeddings, self.labels)
        results = evaluator.run_grid_search(clustering_config)
        return results


# =========================================================================
# REPORT GENERATION
# =========================================================================
def write_report(ver, iden, clust, speed, clustering_results, n_persons):
    """Write both JSON and Markdown reports."""

    all_results = {
        "model": MODEL_NAME,
        "mode": MODE,
        "n_persons": n_persons,
        "distance_metric": "cosine_similarity",
        "verification": ver,
        "identification": iden,
        "clustering_baseline": clust,
        "performance": speed,
        "clustering_benchmark": clustering_results
    }

    # Save JSON report
    with open(REPORT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON report: {REPORT_JSON.name}")

    # Generate Markdown Table for clustering benchmark
    table_md = generate_markdown_table(clustering_results, top_n=20)

    # Save Markdown report
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Report: Model {MODEL_NAME} ({MODE} mode)\n\n")
        f.write(f"- **Model Pack**: {MODEL_NAME}\n")
        f.write(f"- **Extraction Mode**: {MODE}\n")
        f.write(f"- **Dataset size**: {n_persons} unique persons\n")
        f.write("- **Distance Metric**: Cosine Similarity / Distance\n\n")

        # Performance (Speed)
        if speed:
            f.write("## 1. Performance & Speed\n\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            for k, v in speed.items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n")

        # Verification
        f.write("## 2. Face Verification (1:1)\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for k, v in ver.items():
            f.write(f"| {k} | {v} |\n")
        f.write(f"\n![Verification]({MODEL_NAME}_{MODE}_verification.png)\n\n")

        # Identification
        f.write("## 3. Face Identification (1:N)\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for k, v in iden.items():
            if v is not None:
                f.write(f"| {k} | {v} |\n")
        f.write(f"\n![CMC]({MODEL_NAME}_{MODE}_cmc.png)\n\n")

        # Clustering Baseline
        f.write("## 4. Embedding Space Quality (Baseline)\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for k, v in clust.items():
            f.write(f"| {k} | {v} |\n")
        f.write(f"\n![t-SNE]({MODEL_NAME}_{MODE}_tsne.png)\n\n")
        
        # Add UMAP plot reference if file exists
        umap_file = RESULTS_DIR / f"{MODEL_NAME}_{MODE}_umap.png"
        if umap_file.exists():
            f.write(f"![UMAP]({MODEL_NAME}_{MODE}_umap.png)\n\n")

        # Clustering Algorithm Benchmarks
        f.write("## 5. Clustering Algorithm Benchmark (Grid Search)\n\n")
        f.write(
            "The following table compares different clustering algorithms (DBSCAN, HDBSCAN, KMeans, Agglomerative Clustering) "
            "over various parameter configurations. The runs are ranked by **BCubed F1 score** (the harmonic mean of BCubed "
            "Precision and Recall), which evaluates how accurately the algorithm groups faces of the same identity together "
            "without combining different identities.\n\n"
        )
        f.write(table_md)
        f.write("\n\n")
        
        # Best model summary
        if clustering_results:
            best = clustering_results[0]
            f.write("### Best Clustering Configuration\n\n")
            f.write(f"- **Algorithm**: {best['algorithm']}\n")
            f.write(f"- **Parameters**: `{best['param_str']}`\n")
            f.write(f"- **BCubed F1 Score**: **{best['bcubed_f1']:.4f}**\n")
            f.write(f"- **BCubed Precision**: {best['bcubed_precision']:.4f}\n")
            f.write(f"- **BCubed Recall**: {best['bcubed_recall']:.4f}\n")
            f.write(f"- **Pairwise F1 Score**: **{best['pairwise_f1']:.4f}**\n")
            f.write(f"- **Pairwise Precision**: {best['pairwise_precision']:.4f}\n")
            f.write(f"- **Pairwise Recall**: {best['pairwise_recall']:.4f}\n")
            f.write(f"- **Clusters Found**: {best['n_clusters']} (True count: {n_persons})\n")
            f.write(f"- **Noise Ratio**: {best['noise_ratio']*100:.1f}% ({best['n_noise']} items unclustered)\n")

    print(f"Saved Markdown report: {REPORT_MD.name}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings ...")
    embeddings, labels, paths, splits = load_data()
    n_persons = len(np.unique(labels))
    print(f"  Shape:   {embeddings.shape}")
    print(f"  Persons: {n_persons}")
    print(f"  Gallery: {(splits == 'gallery').sum()},  Query: {(splits == 'query').sum()}")

    # 1. Verification Evaluation
    verification_eval = FaceVerificationEval(embeddings, labels, RESULTS_DIR, MODEL_NAME, MODE)
    ver = verification_eval.evaluate()
    
    # 2. Identification Evaluation
    identification_eval = FaceIdentificationEval(embeddings, labels, splits, RESULTS_DIR, MODEL_NAME, MODE)
    iden = identification_eval.evaluate()
    
    # 3. Clustering Baseline (Silhouette, Intra/Inter dist, t-SNE, UMAP)
    clustering_eval = FaceClusteringEval(embeddings, labels, RESULTS_DIR, MODEL_NAME, MODE)
    clust = clustering_eval.evaluate_space_quality(tsne_perplexity=cfg["evaluation"]["tsne_perplexity"])

    # 4. Clustering Algorithm Benchmark (Grid Search)
    clustering_results = clustering_eval.run_grid_search(cfg["clustering"])

    # Load performance/speed metrics
    speed = {}
    if SPEED_PATH.exists():
        sd = np.load(SPEED_PATH)
        speed = {
            "avg_inference_ms": float(sd["avg_inference_ms"]),
            "throughput_ips": float(sd["throughput_ips"]),
            "total_images": int(sd["total_images"]),
            "total_time_s": float(sd["total_time_s"]),
        }

    # 5. Write Report
    write_report(ver, iden, clust, speed, clustering_results, n_persons)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print(f"Results located at: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
