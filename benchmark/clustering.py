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

def run_infomap(embeddings, k=50, min_sim=0.58):
    """
    Runs Infomap face clustering algorithm.
    Compatible with both Infomap 1.x and Infomap 2.x API versions.
    
    :param embeddings: Feature matrix (N, D)
    :param k: Top-K neighbors for KNN graph
    :param min_sim: Minimum cosine similarity threshold for edges
    :return: pred_labels (N,) numpy array
    """
    try:
        import infomap
    except ImportError:
        raise ImportError("The 'infomap' package is required to run Infomap clustering. Please install via: pip install infomap")

    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    emb_norm = emb_norm.astype('float32')
    N, dim = emb_norm.shape

    try:
        import faiss
        index = faiss.IndexFlatIP(dim)
        index.add(emb_norm)
        sims, nbrs = index.search(emb_norm, k=min(k, N))
    except ImportError:
        sim_matrix = emb_norm @ emb_norm.T
        nbrs = np.argsort(-sim_matrix, axis=1)[:, :min(k, N)]
        sims = np.take_along_axis(sim_matrix, nbrs, axis=1)

    singletons = set(range(N))
    links = {}
    for i in range(N):
        for j_idx, sim in zip(nbrs[i], sims[i]):
            if i == j_idx:
                continue
            if sim >= min_sim:
                links[(i, int(j_idx))] = float(sim)
                singletons.discard(i)
                singletons.discard(int(j_idx))

    infomap_wrapper = infomap.Infomap("--two-level --directed")
    
    # --- Add links with API version compatibility ---
    if hasattr(infomap_wrapper, "add_links"):
        formatted_links = [(int(src), int(dst), float(weight)) for (src, dst), weight in links.items()]
        infomap_wrapper.add_links(formatted_links)
    elif hasattr(infomap_wrapper, "add_link"):
        for (src, dst), weight in links.items():
            infomap_wrapper.add_link(int(src), int(dst), float(weight))
    elif hasattr(infomap_wrapper, "addLink"):
        for (src, dst), weight in links.items():
            infomap_wrapper.addLink(int(src), int(dst), float(weight))
    else:
        raise AttributeError("Infomap object has no attribute 'add_links', 'add_link', or 'addLink'")

    # Run community detection
    infomap_wrapper.run()

    # --- Extract cluster labels with API version compatibility ---
    pred_labels = np.full(N, -1, dtype=int)
    
    nodes_collection = None
    if hasattr(infomap_wrapper, "leaves"):
        nodes_collection = infomap_wrapper.leaves
    elif hasattr(infomap_wrapper, "nodes"):
        nodes_collection = infomap_wrapper.nodes
    elif hasattr(infomap_wrapper, "iterTree"):
        nodes_collection = infomap_wrapper.iterTree()
    elif hasattr(infomap_wrapper, "iter_tree"):
        nodes_collection = infomap_wrapper.iter_tree()
    else:
        nodes_collection = infomap_wrapper

    for node in nodes_collection:
        is_leaf_attr = getattr(node, "is_leaf", getattr(node, "isLeaf", True))
        is_leaf = is_leaf_attr() if callable(is_leaf_attr) else is_leaf_attr
        
        if is_leaf:
            node_id = getattr(node, "node_id", getattr(node, "physicalId", getattr(node, "id", None)))
            mod_attr = getattr(node, "module_id", getattr(node, "moduleIndex", None))
            mod_id = mod_attr() if callable(mod_attr) else mod_attr
            
            if node_id is not None and 0 <= node_id < N and mod_id is not None:
                pred_labels[node_id] = int(mod_id)

    # Assign singletons unique cluster IDs
    next_cluster_id = pred_labels.max() + 1 if pred_labels.max() >= 0 else 0
    for node_id in singletons:
        pred_labels[node_id] = next_cluster_id
        next_cluster_id += 1

    return pred_labels


def post_process_merge_clusters(embeddings: np.ndarray, pred_labels: np.ndarray, merge_sim: float = 0.40, max_small_size: int = 3) -> np.ndarray:
    """
    Post-processes clustering output by re-assigning small clusters (size <= max_small_size)
    to the nearest large cluster using Image-to-Image nearest neighbor matching.

    Strategy: Image-level Nearest Neighbor Re-Assignment
      1. Separate clusters into 'large' (size > max_small_size) and 'small' (size <= max_small_size).
      2. Build a FAISS index (or numpy fallback) of ALL images belonging to large clusters.
      3. For each image in a small cluster, find its nearest neighbor image in the large-cluster index.
      4. Determine which large cluster the small cluster should join via majority voting.
      5. Reassign if the best neighbor similarity >= merge_sim.

    :param embeddings: Feature matrix (N, D)
    :param pred_labels: Initial cluster labels (N,)
    :param merge_sim: Min cosine similarity (image-to-image) to reassign (e.g. 0.35 to 0.45)
    :param max_small_size: Maximum cluster size to be considered 'small' and eligible for reassignment
    :return: post_processed_labels (N,) numpy array
    """
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    emb_norm = emb_norm.astype('float32')
    N = len(emb_norm)

    unique_clusters = np.unique(pred_labels)
    valid_clusters = sorted([c for c in unique_clusters if c >= 0])
    n_initial_clusters = len(valid_clusters)

    if n_initial_clusters <= 1:
        return pred_labels.copy()

    # 1. Compute cluster sizes
    cluster_sizes = {}
    for c_id in valid_clusters:
        cluster_sizes[c_id] = int(np.sum(pred_labels == c_id))

    large_clusters = set(c_id for c_id in valid_clusters if cluster_sizes[c_id] > max_small_size)
    small_clusters = [c_id for c_id in valid_clusters if cluster_sizes[c_id] <= max_small_size]

    if not large_clusters or not small_clusters:
        print(f"  [Post-Process] No eligible clusters to merge (large={len(large_clusters)}, small={len(small_clusters)})")
        return pred_labels.copy()

    # 2. Build index of images belonging to large clusters
    large_mask = np.array([lbl in large_clusters for lbl in pred_labels], dtype=bool)
    large_indices = np.where(large_mask)[0]
    large_embeddings = emb_norm[large_indices]
    large_labels = pred_labels[large_indices]

    try:
        import faiss
        dim = large_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(large_embeddings)

        def query_nn(query_embs, k=1):
            sims, idxs = index.search(query_embs, k)
            return sims, idxs
    except ImportError:
        def query_nn(query_embs, k=1):
            sim_matrix = query_embs @ large_embeddings.T
            idxs = np.argsort(-sim_matrix, axis=1)[:, :k]
            sims = np.take_along_axis(sim_matrix, idxs, axis=1)
            return sims, idxs

    # 3. For each small cluster, find nearest large cluster via image-level NN
    reassign_map = {}
    n_reassigned = 0
    n_kept = 0

    for s_id in small_clusters:
        s_mask = (pred_labels == s_id)
        s_indices = np.where(s_mask)[0]
        s_embeddings = emb_norm[s_indices]

        # Find nearest neighbor in large clusters for each image in the small cluster
        sims, nn_idxs = query_nn(s_embeddings, k=1)
        sims = sims.flatten()
        nn_idxs = nn_idxs.flatten()

        # Get the large cluster label for each nearest neighbor
        nn_large_labels = large_labels[nn_idxs]

        # Majority vote: which large cluster do most images point to?
        nn_unique, nn_counts = np.unique(nn_large_labels, return_counts=True)
        best_large_id = nn_unique[np.argmax(nn_counts)]

        # Check similarity: use the max similarity among images pointing to the best large cluster
        best_mask = (nn_large_labels == best_large_id)
        best_sim = float(np.max(sims[best_mask]))

        if best_sim >= merge_sim:
            reassign_map[s_id] = int(best_large_id)
            n_reassigned += 1
        else:
            n_kept += 1

    # 4. Apply reassignment
    post_processed_labels = pred_labels.copy()
    for old_id, new_id in reassign_map.items():
        post_processed_labels[pred_labels == old_id] = new_id

    n_final_clusters = len(np.unique(post_processed_labels[post_processed_labels >= 0]))
    print(f"  [Post-Process] Image-level NN Re-Assignment (merge_sim={merge_sim:.2f}, max_small_size={max_small_size}):")
    print(f"    - Large clusters (>{max_small_size} imgs): {len(large_clusters)}")
    print(f"    - Small clusters (≤{max_small_size} imgs): {len(small_clusters)}")
    print(f"    - Reassigned to large: {n_reassigned} | Kept as-is: {n_kept}")
    print(f"    - Result: {n_initial_clusters} initial clusters --> {n_final_clusters} final clusters")

    return post_processed_labels


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

            elif name == "infomap":
                grid = {
                    "k": alg_cfg.get("k", [50]),
                    "min_sim": alg_cfg.get("min_sim", [0.58])
                }
                keys, values = zip(*grid.items())
                for combination in itertools.product(*values):
                    params = dict(zip(keys, combination))
                    try:
                        labels = run_infomap(self.embeddings, k=params["k"], min_sim=params["min_sim"])
                        all_results.append(self._evaluate_run("Infomap", params, labels))
                    except Exception as e:
                        print(f"    Error running Infomap with {params}: {e}")
                        
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
