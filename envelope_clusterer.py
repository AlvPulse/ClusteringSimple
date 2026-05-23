import numpy as np
import os
import yaml
import csv
import shutil
from datetime import datetime
import wave

class ImprovedRuleEnvelopeClusterer:
    """
    A Rule Envelope Clusterer updated to mitigate the 'Curse of Dimensionality'.
    Rather than multiplying growths or overlaps across all N dimensions (which exponentially shrinks/grows),
    we use 1D max-growth checks and average 1D overlaps.
    It also includes an adaptive proximity fallback via running standard deviations.
    """
    EPS = 1e-6

    def __init__(self, config_path="ClusteringConfig.yaml", adaptive_proximity=True, adaptive_multiplier=0.5):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        c = cfg["clustering"]

        self.max_clusters = c["max_clusters"]
        self.base_proximity = c["feature_proximity"]
        self.mode = c["mode"]
        self.max_growth = c.get("max_growth_ratio", 2.0)
        self.overlap_merge = c.get("overlap_merge_ratio", 0.8)

        self.min_cluster_points = c.get("min_cluster_points", 2)
        self.merge_interval = c.get("merge_interval", 10)
        self.save_state_interval = c.get("save_state_interval_sec", 60)
        self.enable_forgetting = c.get("enable_forgetting", False)
        self.forgetting_factor = c.get("forgetting_factor", 0.995)

        self.clusters = []          # list of envelopes: [{feat: [min, max]}]
        self.cluster_counts = []
        self.cluster_ids = []
        self.cluster_last_update = []
        self.next_cluster_id = 0

        self.log_dir = cfg["logging"]["directory"]
        self.track_dir = cfg["tracking"]["directory"]
        self.keep = cfg["tracking"]["files_per_cluster"]
        self.max_rows = cfg["logging"]["max_rows_per_file"]

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.track_dir, exist_ok=True)

        self.log_file_idx = 0
        self.log_rows = 0
        self.current_log = None

        self.points_processed = 0
        self.last_merge = 0
        self.last_save_time = datetime.now()

        # Adaptive Proximity Tracking (Welford's Algorithm)
        self.adaptive_proximity = adaptive_proximity
        self.adaptive_multiplier = adaptive_multiplier
        self.running_means = {}
        self.running_m2 = {}
        self.n_samples = 0

    def _update_running_stats(self, point):
        self.n_samples += 1
        for feat, val in point.items():
            if feat not in self.running_means:
                self.running_means[feat] = 0.0
                self.running_m2[feat] = 0.0

            delta = val - self.running_means[feat]
            self.running_means[feat] += delta / self.n_samples
            delta2 = val - self.running_means[feat]
            self.running_m2[feat] += delta * delta2

    def _get_proximity(self, feat):
        if self.adaptive_proximity and self.n_samples > 1 and feat in self.running_m2:
            std = np.sqrt(self.running_m2[feat] / self.n_samples)
            if std > self.EPS:
                return std * self.adaptive_multiplier
        # Fallback to config
        return self.base_proximity.get(feat, 0.1)

    def partial_fit(self, x, audio_in=None, sample_rate=None, file=None):
        self._update_running_stats(x)
        self.points_processed += 1

        best_idx = self._find_best_match(x)

        if best_idx is None:
            assigned_id = self._create_cluster(x)
            self.log_cluster(assigned_id)
        else:
            expanded = self._expand_cluster(best_idx, x)
            if expanded:
                self.log_cluster(best_idx)
            assigned_id = best_idx

        self.cluster_counts[assigned_id] += 1

        if self.enable_forgetting:
            self._apply_forgetting()

        if file is not None:
            self.track_file(assigned_id, file)
        elif audio_in is not None and sample_rate is not None:
            self.track_data(cluster_id=assigned_id, audio_data=audio_in, sample_rate=sample_rate)

        cluster_id_to_return = self.cluster_ids[assigned_id]

        if (self.points_processed - self.last_merge) >= self.merge_interval:
            self.merge_clusters()
            self.last_merge = self.points_processed

        if (datetime.now() - self.last_save_time).total_seconds() >= self.save_state_interval:
            self.save_state()
            self.last_save_time = datetime.now()

        return cluster_id_to_return

    def fit_predict(self, X, files=None):
        labels = []
        if files is None:
            files = [None] * len(X)
        for x, f in zip(X, files):
            cid = self.partial_fit(x, f)
            labels.append(cid)

        # Re-map cluster_ids to 0..N consecutive labels for metrics
        unique_ids = list(set(labels))
        id_to_label = {uid: i for i, uid in enumerate(unique_ids)}
        mapped_labels = [id_to_label[l] for l in labels]

        return np.array(mapped_labels)

    def _find_best_match(self, x):
        best_idx = None
        best_dist = float('inf')

        for i, env in enumerate(self.clusters):
            if self._point_in_envelope(x, env):
                # Calculate simple distance to centroid
                dist = 0
                for feat, val in x.items():
                    low, high = env[feat]
                    centroid = (low + high) / 2.0
                    dist += (val - centroid)**2

                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        return best_idx

    def _point_in_envelope(self, point, envelope):
        for feat, val in point.items():
            low, high = envelope[feat]
            p = self._get_proximity(feat)
            # Allow slight padding when checking containment
            if not (low - p - self.EPS <= val <= high + p + self.EPS):
                return False
        return True

    def _expand_cluster(self, idx, point):
        env = self.clusters[idx]
        changed = False

        if self.mode == "controlled":
            # IMPROVEMENT: Check 1D max growth rather than multiplicative growth
            for feat, val in point.items():
                low, high = env[feat]
                old_width = max(high - low, self.EPS)
                p = self._get_proximity(feat)
                new_low = min(low, val - p)
                new_high = max(high, val + p)
                new_width = new_high - new_low

                growth = new_width / old_width
                if growth > self.max_growth:
                    return False  # expansion denied because a single feature grew too much

        # Apply expansion
        for feat, val in point.items():
            p = self._get_proximity(feat)
            new_low = min(env[feat][0], val - p)
            new_high = max(env[feat][1], val + p)
            if new_low != env[feat][0] or new_high != env[feat][1]:
                env[feat] = [new_low, new_high]
                changed = True

        if changed:
            self.cluster_last_update[idx] = datetime.now()
        return changed

    def _create_cluster(self, point):
        if len(self.clusters) >= self.max_clusters:
            min_count = min(self.cluster_counts)
            candidates = [i for i, cnt in enumerate(self.cluster_counts) if cnt == min_count]
            if candidates:
                to_remove = min(candidates, key=lambda i: self.cluster_last_update[i])
                self._remove_cluster(to_remove)

        new_env = {}
        for feat, val in point.items():
            p = self._get_proximity(feat)
            new_env[feat] = [val - p, val + p]

        self.clusters.append(new_env)
        self.cluster_counts.append(1)
        new_id = self.next_cluster_id
        self.cluster_ids.append(new_id)
        self.cluster_last_update.append(datetime.now())
        self.next_cluster_id += 1

        return len(self.clusters) - 1

    def _remove_cluster(self, idx):
        removed_id = self.cluster_ids[idx]
        del self.clusters[idx]
        del self.cluster_counts[idx]
        del self.cluster_ids[idx]
        del self.cluster_last_update[idx]
        self._log_removal_event(removed_id)

    # ----------------------------------------------------
    # Improved Merge Logic (1D Average Overlap)
    # ----------------------------------------------------
    def _calculate_1d_overlap_ratio(self, a, b):
        """Calculates the average 1D overlap ratio across all features to avoid exponential decay."""
        ratios = []
        for feat in a:
            low_a, high_a = a[feat]
            low_b, high_b = b[feat]

            overlap_low = max(low_a, low_b)
            overlap_high = min(high_a, high_b)
            overlap_width = max(0, overlap_high - overlap_low)

            min_width = min(high_a - low_a, high_b - low_b)
            min_width = max(min_width, self.EPS)

            ratios.append(overlap_width / min_width)

        return sum(ratios) / len(ratios)

    def contained(self, a, b):
        for feat in a:
            if not (b[feat][0] <= a[feat][0] and a[feat][1] <= b[feat][1]):
                return False
        return True

    def merge_two(self, i, j):
        a = self.clusters[i]
        b = self.clusters[j]
        for feat in a:
            a[feat][0] = min(a[feat][0], b[feat][0])
            a[feat][1] = max(a[feat][1], b[feat][1])
        self.cluster_counts[i] += self.cluster_counts[j]
        self.cluster_last_update[i] = max(self.cluster_last_update[i], self.cluster_last_update[j])
        self.log_merge_event(self.cluster_ids[i], self.cluster_ids[j])
        del self.clusters[j]
        del self.cluster_counts[j]
        del self.cluster_ids[j]
        del self.cluster_last_update[j]

    def merge_clusters(self):
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(self.clusters):
                j = i + 1
                while j < len(self.clusters):
                    a = self.clusters[i]
                    b = self.clusters[j]
                    do_merge = False

                    if self.contained(a, b) or self.contained(b, a):
                        do_merge = True
                    elif self.mode == "controlled":
                        avg_overlap = self._calculate_1d_overlap_ratio(a, b)
                        if avg_overlap > self.overlap_merge:
                            do_merge = True

                    if do_merge:
                        self.merge_two(i, j)
                        changed = True
                    else:
                        j += 1
                i += 1

    def _apply_forgetting(self):
        # Implementation for forgetting if enabled
        pass

    # -- Basic Logging implementations --
    def track_file(self, cluster_id, file): pass
    def log_cluster(self, idx): pass
    def save_state(self): pass
    def log_merge_event(self, id_a, id_b): pass
    def _log_removal_event(self, removed_id): pass
    def track_data(self, cluster_id, audio_data, sample_rate): pass