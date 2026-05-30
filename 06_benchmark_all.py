import os
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch
from utils import load_data, evaluate_clustering, generate_html_report, MinesweeperStreamingClustering
from envelope_clusterer import ImprovedRuleEnvelopeClusterer

def evaluate_and_report(name, model, X, y_true, files, class_names, is_envelope=False):
    print(f"Running {name}...")

    if is_envelope:
        feature_names = [
            'dominant_freq', 'rms_energy',
            'low_band_ratio', 'mid_band_ratio', 'high_band_ratio',
            'spectral_flatness', 'spectral_entropy', 'spectral_kurtosis',
            'spectral_centroid', 'temporal_entropy',
            'zcr_mean', 'zcr_std', 'harmonic_score'
        ]
        X_dicts = [{feat: val for feat, val in zip(feature_names, row)} for row in X]
        # Ignore track writing during generic benchmark by passing files=None
        y_pred = model.fit_predict(X_dicts, files=None)
    else:
        y_pred = model.fit_predict(X)

    metrics = evaluate_clustering(y_true, y_pred)

    generate_html_report(
        f"06_Benchmark_{name.replace(' ', '_')}",
        files, y_true, y_pred, class_names, metrics,
        name
    )

    return metrics, len(set(y_pred))

def main():
    print("Loading data with full 13 expanded features...\n")
    X, y_true, files, class_names = load_data("dummy_audio", use_all_features=True)
    n_classes = len(class_names)

    algorithms = {
        "K-Means (Batch Centroid)": KMeans(n_clusters=n_classes, random_state=42, n_init=10),
        "MiniBatchKMeans (Streaming Centroid)": MiniBatchKMeans(n_clusters=n_classes, random_state=42, n_init=10),
        "DBSCAN (Density-Based)": DBSCAN(eps=2.5, min_samples=3),
        "Birch (Tree-Based)": Birch(n_clusters=n_classes, threshold=1.5),
        "Minesweeper (Custom Streaming)": MinesweeperStreamingClustering(distance_threshold=3.0, max_clusters=n_classes)
    }

    results = {}

    for name, model in algorithms.items():
        metrics, n_clusters = evaluate_and_report(name, model, X, y_true, files, class_names)
        metrics["Clusters"] = n_clusters
        results[name] = metrics

    # Envelope clusterer needs special handling
    name = "Rule Envelope (Custom Streaming)"
    env_clusterer = ImprovedRuleEnvelopeClusterer(config_path="ClusteringConfig.yaml")
    metrics, n_clusters = evaluate_and_report(name, env_clusterer, X, y_true, files, class_names, is_envelope=True)
    metrics["Clusters"] = n_clusters
    results[name] = metrics

    # Print summary table
    print("\n" + "="*85)
    print(f"{'Algorithm':<40} | {'Clusters':<8} | {'Purity':<8} | {'ARI':<8} | {'NMI':<8}")
    print("-" * 85)

    for name, m in results.items():
        print(f"{name:<40} | {m['Clusters']:<8} | {m['Purity']:<8.4f} | {m['ARI']:<8.4f} | {m['NMI']:<8.4f}")

    print("="*85)
    print("\nCheck the generated HTML reports to listen to the clustered audio distributions!")

if __name__ == "__main__":
    main()