import os
import time
import numpy as np
from utils import load_data
from envelope_clusterer import ImprovedRuleEnvelopeClusterer

def main():
    print("Initializing Long-Running Streaming Simulation...")

    # Use full features
    X, y_true, files, class_names = load_data("dummy_audio", use_all_features=True)

    # Ensure our feature ordering matches utils.py extraction
    feature_names = [
        'dominant_freq', 'rms_energy',
        'low_band_ratio', 'mid_band_ratio', 'high_band_ratio',
        'spectral_flatness', 'spectral_entropy', 'spectral_kurtosis',
        'spectral_centroid', 'temporal_entropy',
        'zcr_mean', 'zcr_std', 'harmonic_score'
    ]

    # In order to simulate a long running stream, we will loop over the dummy data
    # multiple times, adding slight jitter to simulate continuous input.
    X_dicts = []

    # Let's create a stream of 1500 points (5 loops over the 300 dummy samples)
    for loop in range(5):
        for i, row in enumerate(X):
            # Add small random noise to prevent identical points
            jittered_row = row + np.random.normal(0, 0.05, size=row.shape)
            point = {feat: val for feat, val in zip(feature_names, jittered_row)}
            X_dicts.append((point, files[i]))

    # We use a custom save interval for testing so we don't have to wait 60 seconds
    clusterer = ImprovedRuleEnvelopeClusterer(config_path="ClusteringConfig.yaml")
    clusterer.save_state_interval = 2 # Save every 2 seconds for this simulation

    print(f"Streaming {len(X_dicts)} data points (simulating hours of audio)...")

    start_time = time.time()
    for i, (point_dict, file_path) in enumerate(X_dicts):
        # We pass the file_path to trigger track_file, testing storage limits
        # We pass a dummy sample rate since we're using file tracking instead of raw audio tracking
        clusterer.partial_fit(point_dict, file=file_path)

        # Add a tiny sleep to simulate streaming and trigger time-based saving
        time.sleep(0.01)

        if (i+1) % 100 == 0:
            print(f"Processed {i+1} points. Active clusters: {len(clusterer.clusters)}")

    end_time = time.time()
    print(f"\nSimulation complete in {end_time - start_time:.2f} seconds.")
    print("Check the 'logs/' folder for 'event_logs.csv' and 'feature_importance.csv'.")
    print("Check the 'tracked_samples/' folder to confirm storage limits are respected.")

if __name__ == "__main__":
    main()