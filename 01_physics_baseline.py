import os
from sklearn.cluster import KMeans
from utils import load_data, evaluate_clustering, MinesweeperStreamingClustering, generate_html_report

def main():
    print("Loading data with 2 highly interpretable baseline features...")
    X, y_true, files, class_names = load_data("dummy_audio", use_all_features=False)

    # 1. Traditional SOTA (K-Means)
    print("\nRunning K-Means (SOTA Baseline)...")
    kmeans = KMeans(n_clusters=len(class_names), random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)

    metrics_kmeans = evaluate_clustering(y_true, y_kmeans)
    print("K-Means Metrics:", metrics_kmeans)

    generate_html_report(
        "01_Physics_Baseline_KMeans",
        files, y_true, y_kmeans, class_names, metrics_kmeans,
        "K-Means (Batch)"
    )

    # 2. Custom Streaming Clustering
    print("\nRunning Custom Streaming Clustering...")
    streamer = MinesweeperStreamingClustering(distance_threshold=1.5, max_clusters=len(class_names))
    y_stream = streamer.fit_predict(X)

    metrics_stream = evaluate_clustering(y_true, y_stream)
    print("Streaming Metrics:", metrics_stream)

    generate_html_report(
        "01_Physics_Baseline_Streaming",
        files, y_true, y_stream, class_names, metrics_stream,
        "Custom Streaming (Minesweeper)"
    )

if __name__ == "__main__":
    main()