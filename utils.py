import os
import glob
import numpy as np
import librosa
from sklearn import metrics
from collections import defaultdict
import base64

# --- Metrics ---

def purity_score(y_true, y_pred):
    """Compute contingency matrix (also called confusion matrix)"""
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def evaluate_clustering(y_true, y_pred):
    purity = purity_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    return {"Purity": purity, "ARI": ari, "NMI": nmi}

# --- Feature Extraction ---

def extract_features(filepath, use_all=False):
    y, sr = librosa.load(filepath, sr=None)

    # Pre-compute
    S = np.abs(librosa.stft(y))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # 2 Interpretable baseline features:
    # 1. Dominant Frequency
    freqs = librosa.fft_frequencies(sr=sr)
    dominant_freq = freqs[np.argmax(np.mean(S, axis=1))]
    # 2. RMS Energy
    rms_energy = np.mean(librosa.feature.rms(y=y))

    features = [dominant_freq, rms_energy]

    if use_all:
        # Extra 11 features to make 13
        # Bands
        band1 = np.sum(S[(freqs > 0) & (freqs <= 1000), :])
        band2 = np.sum(S[(freqs > 1000) & (freqs <= 4000), :])
        band3 = np.sum(S[(freqs > 4000), :])
        total = np.sum(S) + 1e-6
        low_band_ratio = band1 / total
        mid_band_ratio = band2 / total
        high_band_ratio = band3 / total

        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        spectral_entropy = -np.sum((S / total) * np.log(S / total + 1e-6)) / S.size

        # approximate kurtosis over the spectrum
        spectral_kurtosis = np.mean(np.mean(S**4, axis=0) / (np.mean(S**2, axis=0)**2 + 1e-6))

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # Temporal entropy
        energy_env = librosa.feature.rms(y=y)[0]
        total_env = np.sum(energy_env) + 1e-6
        temporal_entropy = -np.sum((energy_env / total_env) * np.log(energy_env / total_env + 1e-6))

        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # Harmonic score (using HPSS)
        H, P = librosa.effects.hpss(y)
        harmonic_score = np.sum(H**2) / (np.sum(H**2) + np.sum(P**2) + 1e-6)

        features.extend([
            low_band_ratio, mid_band_ratio, high_band_ratio,
            spectral_flatness, spectral_entropy, spectral_kurtosis,
            spectral_centroid, temporal_entropy, zcr_mean, zcr_std,
            harmonic_score
        ])

    return np.array(features)

# --- Data Loading ---

def load_data(base_dir, use_all_features=False):
    files = []
    labels = []
    features_list = []

    class_names = sorted(os.listdir(base_dir))
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for f in glob.glob(os.path.join(class_dir, "*.wav")):
            feats = extract_features(f, use_all=use_all_features)
            files.append(f)
            labels.append(i)
            features_list.append(feats)

    # standardize features
    X = np.array(features_list)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0) + 1e-6
    X = (X - means) / stds

    return X, np.array(labels), files, class_names

# --- Custom Streaming Clustering Algorithm ---

class MinesweeperStreamingClustering:
    """
    A simple custom streaming clustering algorithm.
    It reads samples one by one.
    If a sample is within a distance threshold to a cluster, it assigns it to the cluster
    and updates the cluster centroid (pulling it closer).
    If it is too far from all clusters, it creates a new cluster.
    """
    def __init__(self, distance_threshold=2.0, max_clusters=10):
        self.distance_threshold = distance_threshold
        self.max_clusters = max_clusters
        self.centroids = []
        self.cluster_counts = []

    def fit_predict(self, X):
        labels = []
        for x in X:
            if len(self.centroids) == 0:
                self.centroids.append(x)
                self.cluster_counts.append(1)
                labels.append(0)
            else:
                # Find nearest centroid
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                min_dist = min(distances)
                closest_idx = np.argmin(distances)

                if min_dist < self.distance_threshold or len(self.centroids) >= self.max_clusters:
                    # Assign to closest
                    labels.append(closest_idx)
                    # Update centroid (streaming update)
                    n = self.cluster_counts[closest_idx]
                    self.centroids[closest_idx] = (self.centroids[closest_idx] * n + x) / (n + 1)
                    self.cluster_counts[closest_idx] += 1
                else:
                    # Create new cluster
                    self.centroids.append(x)
                    self.cluster_counts.append(1)
                    labels.append(len(self.centroids) - 1)
        return np.array(labels)

# --- Reporting ---

def generate_html_report(report_name, files, ground_truth, predictions, class_names, metrics_dict, algo_name):
    html = f"<html><head><title>{report_name}</title></head><body>"
    html += f"<h1>{report_name}</h1>"
    html += f"<h2>Algorithm: {algo_name}</h2>"

    html += "<h3>Metrics</h3><ul>"
    for k, v in metrics_dict.items():
        html += f"<li><b>{k}:</b> {v:.4f}</li>"
    html += "</ul>"

    html += "<h3>Cluster Audios</h3>"

    # Group by prediction
    clusters = defaultdict(list)
    for f, true_label, pred_label in zip(files, ground_truth, predictions):
        clusters[pred_label].append((f, class_names[true_label]))

    for c_id in sorted(clusters.keys()):
        html += f"<h4>Cluster {c_id}</h4>"
        html += "<ul>"

        # Limit to 5 per cluster for backtracking
        for f, true_class in clusters[c_id][:5]:
            # Embed audio as base64 so report is standalone
            with open(f, "rb") as audio_file:
                audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")

            html += "<li>"
            html += f"Ground Truth: <b>{true_class}</b> <br>"
            html += f"<audio controls><source src='data:audio/wav;base64,{audio_b64}' type='audio/wav'></audio>"
            html += "</li>"

        if len(clusters[c_id]) > 5:
            html += f"<li>... and {len(clusters[c_id]) - 5} more files.</li>"
        html += "</ul>"

    html += "</body></html>"

    out_file = f"report_{report_name.replace(' ', '_').lower()}.html"
    with open(out_file, "w") as f:
        f.write(html)
    print(f"Report saved to {out_file}")