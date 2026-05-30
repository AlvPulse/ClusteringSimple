import os
from sklearn.cluster import KMeans
from utils import load_data, evaluate_clustering, MinesweeperStreamingClustering, generate_html_report

def main():
    print("Loading data with full 13 expanded features...")
    X, y_true, files, class_names = load_data("dummy_audio", use_all_features=True)

    # 1. Traditional SOTA (K-Means)
    print("\nRunning K-Means (SOTA Baseline)...")
    kmeans = KMeans(n_clusters=len(class_names), random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)

    metrics_kmeans = evaluate_clustering(y_true, y_kmeans)
    print("K-Means Metrics:", metrics_kmeans)

    generate_html_report(
        "02_Expanded_Features_KMeans",
        files, y_true, y_kmeans, class_names, metrics_kmeans,
        "K-Means (Batch) - 13 Features"
    )

    # 2. Custom Streaming Clustering
    print("\nRunning Custom Streaming Clustering...")
    # Increase threshold due to higher dimensionality
    streamer = MinesweeperStreamingClustering(distance_threshold=3.0, max_clusters=len(class_names))
    y_stream = streamer.fit_predict(X)

    metrics_stream = evaluate_clustering(y_true, y_stream)
    print("Streaming Metrics:", metrics_stream)

    generate_html_report(
        "02_Expanded_Features_Streaming",
        files, y_true, y_stream, class_names, metrics_stream,
        "Custom Streaming (Minesweeper) - 13 Features"
    )

if __name__ == "__main__":
    main()