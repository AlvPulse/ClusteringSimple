# Audio Clustering Benchmark

This repository demonstrates an audio clustering benchmark, showing the improvement of clustering performance from simple interpretable features to state-of-the-art (SOTA) complex embeddings, while also comparing traditional K-Means with a custom streaming clustering algorithm.

## Motivation
The goal is to show how much streaming clustering can approximate a human baseline (defined by subfolders containing specific classes of audio), while understanding that streaming performance is intrinsically dependent on the order of incoming data.

We use a step-by-step incremental build-up to gain trust in the benchmark:
1. **Physics Baseline (`01_physics_baseline.py`)**: Uses only two highly interpretable features (e.g., dominant frequency and RMS energy).
2. **Expanded Features (`02_expanded_features.py`)**: Explores 13 physics-based and statistical features.
3. **Complex Embeddings (`03_complex_embeddings.py`)**: Uses YAMNet embeddings from TensorFlow Hub to show SOTA feature representations.

## Algorithms
We benchmark the following unsupervised algorithms:
- **K-Means**: The traditional batch-mode SOTA baseline.
- **Custom Streaming ("Minesweeper")**: A custom, incremental streaming clustering that dynamically changes cluster boundaries. As each sample comes in, if it's near an existing cluster, it's absorbed and the cluster may expand. If it is far, it creates a new cluster and acts to limit the boundaries of the others.

## Evaluation Metrics
We evaluate the performance using three distinct metrics:
- **Clustering Purity**: The primary baseline metric. It measures the extent to which clusters contain a single class. Ranging from 0 to 1, higher is better.
- **Adjusted Rand Index (ARI)**: A measure of the similarity between two data clusterings. It adjusts for chance grouping. A score of 1.0 means perfect alignment, while 0 means random.
- **Normalized Mutual Information (NMI)**: An information-theoretic measure that quantifies the mutual dependence between the cluster assignments and the ground truth labels. It normalizes the score between 0 and 1.

## Getting Started
1. Run `python 00_generate_dummy_data.py` to create synthetic test audio files.
2. Run `python 01_physics_baseline.py`, followed by `02` and `03` to observe the benchmarking progression.
3. Open the generated HTML reports (e.g., `report_01_physics_baseline.html`) to back-track and listen to the grouped clusters!