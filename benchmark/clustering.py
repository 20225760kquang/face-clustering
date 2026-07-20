import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from collections import defaultdict
import itertools

def bcubed_metrics(true_labels, pred_labels):
    """
    Computes BCubed Precision, Recall, and F1 score.
    Uses group set intersections for O(N) memory and fast execution.
    """
    N = len(true_labels)
    if N == 0:
        return 0.0, 0.0, 0.0
        
    true_groups = defaultdict(set)
    pred_groups = defaultdict(set)
    for idx, (t, p) in enumerate(zip(true_labels, pred_labels)):
        true_groups[t].add(idx)
        pred_groups[p].add(idx)
        
    precisions = []
    recalls = []
    
    for idx, (t, p) in enumerate(zip(true_labels, pred_labels)):
        t_group = true_groups[t]
        p_group = pred_groups[p]
        
        intersect_size = len(t_group & p_group)
        
        precisions.append(intersect_size / len(p_group))
        recalls.append(intersect_size / len(t_group))
        
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    
    if avg_precision + avg_recall > 0:
        f1 = float(2 * avg_precision * avg_recall / (avg_precision + avg_recall))
    else:
        f1 = 0.0
        
    return float(avg_precision), float(avg_recall), f1


def pairwise_metrics(true_labels, pred_labels):
    """
    Computes Pairwise Precision, Recall, and F1 score.
    Uses contingency matrix for fast execution.
    """
    from sklearn.metrics.cluster import contingency_matrix
    import numpy as np

    cont = contingency_matrix(true_labels, pred_labels)
    
    # TP: pairs in the same class and in the same cluster
    tp = np.sum(cont * (cont - 1)) / 2.0
    
    # Ground truth positives (TP + FN): pairs in the same class in ground truth
    true_sums = np.sum(cont, axis=1)
    gt_positives = np.sum(true_sums * (true_sums - 1)) / 2.0
    
    # Predicted positives (TP + FP): pairs in the same cluster in predictions
    pred_sums = np.sum(cont, axis=0)
    pred_positives = np.sum(pred_sums * (pred_sums - 1)) / 2.0
    
    # Precision = TP / (TP + FP)
    precision = float(tp / pred_positives) if pred_positives > 0 else 0.0
    # Recall = TP / (TP + FN)
    recall = float(tp / gt_positives) if gt_positives > 0 else 0.0
    
    if precision + recall > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = 0.0
        
    return precision, recall, f1

class ClusteringEvaluator:
    def __init__(self, embeddings, true_labels):
        self.embeddings = embeddings
        self.true_labels = true_labels
        self.unique_true_labels = np.unique(true_labels)
        self.n_persons = len(self.unique_true_labels)
        
        # Normalize embeddings to compute cosine distance matrix
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        self.cosine_dist_matrix = 1.0 - np.clip(emb_norm @ emb_norm.T, -1.0, 1.0)
        
    def run_grid_search(self, clustering_config):
        """Runs hyperparameter grid search for all enabled clustering algorithms."""
        all_results = []
        algorithms = clustering_config.get("algorithms", {})
        
        for name, alg_cfg in algorithms.items():
            if not alg_cfg.get("enabled", False):
                continue
                
            print(f"  Running grid search for {name.upper()} ...")
            
            if name == "dbscan":
                grid = {
                    "eps": alg_cfg.get("eps", [0.5]),
                    "min_samples": alg_cfg.get("min_samples", [2]),
                    "metric": alg_cfg.get("metric", ["cosine"])
                }
                keys, values = zip(*grid.items())
                for combination in itertools.product(*values):
                    params = dict(zip(keys, combination))
                    
                    try:
                        if params["metric"] == "cosine":
                            clt = DBSCAN(eps=params["eps"], min_samples=params["min_samples"], metric="precomputed")
                            labels = clt.fit_predict(self.cosine_dist_matrix)
                        else:
                            clt = DBSCAN(eps=params["eps"], min_samples=params["min_samples"], metric=params["metric"])
                            labels = clt.fit_predict(self.embeddings)
                            
                        all_results.append(self._evaluate_run("DBSCAN", params, labels))
                    except Exception as e:
                        print(f"    Error running DBSCAN with {params}: {e}")
                        
            elif name == "hdbscan":
                grid = {
                    "min_cluster_size": alg_cfg.get("min_cluster_size", [5]),
                    "min_samples": alg_cfg.get("min_samples", [1]),
                    "cluster_selection_epsilon": alg_cfg.get("cluster_selection_epsilon", [0.0]),
                    "metric": alg_cfg.get("metric", ["euclidean"])
                }
                keys, values = zip(*grid.items())
                for combination in itertools.product(*values):
                    params = dict(zip(keys, combination))
                    
                    try:
                        clt = HDBSCAN(
                            min_cluster_size=params["min_cluster_size"],
                            min_samples=params["min_samples"],
                            cluster_selection_epsilon=params["cluster_selection_epsilon"],
                            metric=params["metric"]
                        )
                        labels = clt.fit_predict(self.embeddings)
                        all_results.append(self._evaluate_run("HDBSCAN", params, labels))
                    except Exception as e:
                        print(f"    Error running HDBSCAN with {params}: {e}")
                        
            elif name == "kmeans":
                n_clusters_list = alg_cfg.get("n_clusters", ["n_persons"])
                resolved_clusters = []
                for val in n_clusters_list:
                    if val == "n_persons":
                        resolved_clusters.append(self.n_persons)
                    else:
                        resolved_clusters.append(int(val))
                        
                for k in resolved_clusters:
                    params = {"n_clusters": k}
                    try:
                        clt = KMeans(n_clusters=k, random_state=42, n_init="auto")
                        labels = clt.fit_predict(self.embeddings)
                        all_results.append(self._evaluate_run("KMeans", params, labels))
                    except Exception as e:
                        print(f"    Error running KMeans with {params}: {e}")
                        
            elif name == "agglomerative":
                grid = {
                    "distance_threshold": alg_cfg.get("distance_threshold", [None]),
                    "n_clusters": alg_cfg.get("n_clusters", [None]),
                    "linkage": alg_cfg.get("linkage", ["average"]),
                    "metric": alg_cfg.get("metric", ["cosine"])
                }
                keys, values = zip(*grid.items())
                for combination in itertools.product(*values):
                    params = dict(zip(keys, combination))
                    
                    dist_threshold = params["distance_threshold"]
                    n_cl = params["n_clusters"]
                    linkage = params["linkage"]
                    metric = params["metric"]
                    
                    # Handle constraints
                    if dist_threshold is not None:
                        n_cl = None
                    elif n_cl == "n_persons":
                        n_cl = self.n_persons
                        
                    if linkage == "ward":
                        metric = "euclidean"
                        
                    try:
                        if metric == "cosine" and linkage in ["average", "complete", "single"]:
                            clt = AgglomerativeClustering(
                                n_clusters=n_cl,
                                distance_threshold=dist_threshold,
                                linkage=linkage,
                                metric="precomputed"
                            )
                            labels = clt.fit_predict(self.cosine_dist_matrix)
                        else:
                            clt = AgglomerativeClustering(
                                n_clusters=n_cl,
                                distance_threshold=dist_threshold,
                                linkage=linkage,
                                metric=metric
                            )
                            labels = clt.fit_predict(self.embeddings)
                            
                        all_results.append(self._evaluate_run("Agglomerative", params, labels))
                    except Exception as e:
                        print(f"    Error running Agglomerative with {params}: {e}")
                        
        # Sort results by BCubed F1 score descending
        all_results = sorted(all_results, key=lambda x: x["bcubed_f1"], reverse=True)
        return all_results
        
    def _evaluate_run(self, algo_name, params, pred_labels):
        """Computes all metrics for a single clustering run."""
        unique_preds = np.unique(pred_labels)
        n_clusters = int(np.sum(unique_preds >= 0))  # Exclude noise (-1)
        noise_mask = pred_labels == -1
        n_noise = int(np.sum(noise_mask))
        noise_ratio = float(n_noise / len(pred_labels)) if len(pred_labels) > 0 else 0.0
        
        # Calculate standard cluster metrics
        p, r, f1 = bcubed_metrics(self.true_labels, pred_labels)
        pair_p, pair_r, pair_f1 = pairwise_metrics(self.true_labels, pred_labels)
        ari = float(adjusted_rand_score(self.true_labels, pred_labels))
        nmi = float(normalized_mutual_info_score(self.true_labels, pred_labels))
        ami = float(adjusted_mutual_info_score(self.true_labels, pred_labels))
        
        # Parameters to string
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        
        return {
            "algorithm": algo_name,
            "params": params,
            "param_str": param_str,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": noise_ratio,
            "bcubed_precision": p,
            "bcubed_recall": r,
            "bcubed_f1": f1,
            "pairwise_precision": pair_p,
            "pairwise_recall": pair_r,
            "pairwise_f1": pair_f1,
            "ari": ari,
            "nmi": nmi,
            "ami": ami
        }

def generate_markdown_table(results, top_n=20):
    """Generates a formatted Markdown table comparing the top N clustering runs."""
    lines = [
        "| Rank | Algorithm | Parameters | Clusters | Noise % | BCubed F1 | BCubed P | BCubed R | Pairwise F1 | Pairwise P | Pairwise R | ARI | NMI |",
        "|------|-----------|------------|----------|---------|-----------|----------|----------|-------------|------------|------------|-----|-----|"
    ]
    for idx, r in enumerate(results[:top_n], 1):
        noise_pct = f"{r['noise_ratio'] * 100:.1f}%"
        lines.append(
            f"| {idx} | {r['algorithm']} | {r['param_str']} | {r['n_clusters']} | {noise_pct} | "
            f"{r['bcubed_f1']:.4f} | {r['bcubed_precision']:.4f} | {r['bcubed_recall']:.4f} | "
            f"{r['pairwise_f1']:.4f} | {r['pairwise_precision']:.4f} | {r['pairwise_recall']:.4f} | "
            f"{r['ari']:.4f} | {r['nmi']:.4f} |"
        )
    return "\n".join(lines)
