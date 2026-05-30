import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.cluster import KMeans
from utils import evaluate_clustering, MinesweeperStreamingClustering, generate_html_report

# Load YAMNet from TensorFlow Hub
# Using a specific version to ensure stability
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'

def extract_yamnet_embeddings(model, filepath):
    # YAMNet requires audio to be mono, 16kHz
    y, sr = librosa.load(filepath, sr=16000, mono=True)

    # Run the model, get the embeddings
    scores, embeddings, spectrogram = model(y)

    # YAMNet returns embeddings for each frame (0.48s chunks).
    # We average the embeddings over time to get a single vector for the file.
    mean_embedding = np.mean(embeddings.numpy(), axis=0)
    return mean_embedding

def load_data_complex(base_dir):
    print("Loading YAMNet model from TensorFlow Hub...")
    model = hub.load(YAMNET_MODEL_HANDLE)
    print("Model loaded.")

    files = []
    labels = []
    features_list = []

    class_names = sorted(os.listdir(base_dir))
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for f in glob.glob(os.path.join(class_dir, "*.wav")):
            feats = extract_yamnet_embeddings(model, f)
            files.append(f)
            labels.append(i)
            features_list.append(feats)

    # standardize features
    X = np.array(features_list)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0) + 1e-6
    X = (X - means) / stds

    return X, np.array(labels), files, class_names

def main():
    print("Loading data using Complex Embeddings (YAMNet)...")
    X, y_true, files, class_names = load_data_complex("dummy_audio")

    # 1. Traditional SOTA (K-Means)
    print("\nRunning K-Means (SOTA Baseline)...")
    kmeans = KMeans(n_clusters=len(class_names), random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)

    metrics_kmeans = evaluate_clustering(y_true, y_kmeans)
    print("K-Means Metrics:", metrics_kmeans)

    generate_html_report(
        "03_Complex_Embeddings_KMeans",
        files, y_true, y_kmeans, class_names, metrics_kmeans,
        "K-Means (Batch) - YAMNet Embeddings"
    )

    # 2. Custom Streaming Clustering
    print("\nRunning Custom Streaming Clustering...")
    # High dimensionality (1024), we need a larger distance threshold
    streamer = MinesweeperStreamingClustering(distance_threshold=25.0, max_clusters=len(class_names))
    y_stream = streamer.fit_predict(X)

    metrics_stream = evaluate_clustering(y_true, y_stream)
    print("Streaming Metrics:", metrics_stream)

    generate_html_report(
        "03_Complex_Embeddings_Streaming",
        files, y_true, y_stream, class_names, metrics_stream,
        "Custom Streaming (Minesweeper) - YAMNet Embeddings"
    )

if __name__ == "__main__":
    main()