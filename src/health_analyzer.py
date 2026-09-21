"""Health degradation and anomaly analysis module for unsupervised vibration clustering.

Provides mathematical formulation for baseline identification, distance-based
degradation scoring, cluster characterization, and machine health status determination.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class HealthDegradationAnalyzer:
    """Evaluates machine health degradation relative to unsupervised baseline modes."""

    def __init__(
        self,
        baseline_method: str = "earliest",
        baseline_time_fraction: float = 0.25,
        threshold_warning: float = 40.0,
        threshold_critical: float = 75.0,
        smoothing_window: int = 5,
    ):
        self.baseline_method = baseline_method
        self.baseline_time_fraction = baseline_time_fraction
        self.threshold_warning = threshold_warning
        self.threshold_critical = threshold_critical
        self.smoothing_window = smoothing_window

        self.nominal_cluster_id: Optional[int] = None
        self.nominal_center: Optional[np.ndarray] = None
        self.baseline_distance_scale: float = 1.0
        self.cluster_health_profiles: Dict[int, Dict[str, Any]] = {}
        self.is_calibrated: bool = False

    def calibrate(
        self,
        X_scaled: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray,
    ) -> "HealthDegradationAnalyzer":
        """Calibrate nominal baseline state from training feature vectors and cluster labels.

        Assumptions:
        1. Industrial machines start in a healthy nominal baseline state.
        2. The cluster dominant during initial operation (or largest by sample size)
           defines the Nominal Operational Mode.
        3. Feature drift away from this mode signifies mechanical degradation.
        """
        n_samples = len(X_scaled)
        if n_samples == 0 or len(cluster_centers) == 0:
            raise ValueError("Cannot calibrate health analyzer on empty data or clusters.")

        # Determine nominal cluster
        if self.baseline_method == "earliest":
            initial_cutoff = max(1, int(n_samples * self.baseline_time_fraction))
            initial_labels = cluster_labels[:initial_cutoff]
            # Exclude noise (-1) if present
            valid_initial = initial_labels[initial_labels >= 0]
            if len(valid_initial) > 0:
                values, counts = np.unique(valid_initial, return_counts=True)
                self.nominal_cluster_id = int(values[np.argmax(counts)])
            else:
                self.nominal_cluster_id = 0
        else:  # "largest"
            valid_labels = cluster_labels[cluster_labels >= 0]
            if len(valid_labels) > 0:
                values, counts = np.unique(valid_labels, return_counts=True)
                self.nominal_cluster_id = int(values[np.argmax(counts)])
            else:
                self.nominal_cluster_id = 0

        self.nominal_center = cluster_centers[self.nominal_cluster_id].copy()

        # Compute distances of all nominal points to the nominal center to establish reference variance
        nominal_mask = cluster_labels == self.nominal_cluster_id
        if np.any(nominal_mask):
            nominal_points = X_scaled[nominal_mask]
            nominal_dists = np.linalg.norm(nominal_points - self.nominal_center, axis=1)
            # 95th percentile distance represents normal operating variation
            d95 = float(np.percentile(nominal_dists, 95))
            # Scale factor: distance at which score reaches warning threshold (40.0)
            self.baseline_distance_scale = max(d95, 1e-3)
        else:
            self.baseline_distance_scale = 1.0

        # Profile all discovered clusters
        self.cluster_health_profiles = {}
        unique_clusters = np.unique(cluster_labels)

        for c_id in unique_clusters:
            if c_id == -1:
                # Noise points
                self.cluster_health_profiles[-1] = {
                    "cluster_id": -1,
                    "label": "Noise / Severe Outlier",
                    "status": "CRITICAL",
                    "sample_count": int(np.sum(cluster_labels == -1)),
                    "centroid_distance_to_baseline": 999.0,
                    "mean_degradation_score": 95.0,
                }
                continue

            center_c = cluster_centers[c_id]
            dist_to_nominal = float(np.linalg.norm(center_c - self.nominal_center))
            pts_c = X_scaled[cluster_labels == c_id]
            sample_count = int(len(pts_c))

            # Raw score based on normalized distance to nominal mode
            normalized_dist = dist_to_nominal / (self.baseline_distance_scale + 1e-9)
            raw_cluster_score = (normalized_dist / 3.0) * 40.0
            cluster_score = min(100.0, max(0.0, raw_cluster_score))

            if c_id == self.nominal_cluster_id:
                status = "NORMAL"
                desc = "Nominal Healthy Operating Mode"
                cluster_score = 10.0
            elif cluster_score < self.threshold_warning:
                status = "NORMAL"
                desc = "Normal Secondary Operating Regime"
            elif cluster_score < self.threshold_critical:
                status = "WARNING"
                desc = "Incipient Mechanical Degradation (Early Fault Signs)"
            else:
                status = "CRITICAL"
                desc = "Severe Machine Health Degradation"

            self.cluster_health_profiles[int(c_id)] = {
                "cluster_id": int(c_id),
                "description": desc,
                "status": status,
                "sample_count": sample_count,
                "centroid_distance_to_baseline": round(dist_to_nominal, 4),
                "mean_degradation_score": round(cluster_score, 2),
            }

        self.is_calibrated = True
        print(f"[HealthAnalyzer] Nominal cluster identified: Cluster #{self.nominal_cluster_id}")
        print(f"[HealthAnalyzer] Baseline 95% distance scale: {self.baseline_distance_scale:.4f}")
        return self

    def compute_degradation_scores(
        self,
        X_scaled: np.ndarray,
        cluster_labels: np.ndarray,
        smooth: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Calculate continuous degradation scores and health statuses for each observation.

        Returns
        -------
        raw_scores : np.ndarray
            Instantaneous distance-based degradation scores (0 to 100).
        smoothed_scores : np.ndarray
            Rolling-averaged scores to filter instantaneous shock spikes.
        health_statuses : list of str
            Assigned machine status: NORMAL, WARNING, or CRITICAL.
        """
        if not self.is_calibrated or self.nominal_center is None:
            raise RuntimeError("HealthDegradationAnalyzer must be calibrated before computing scores.")

        # Compute Euclidean distance from each sample's feature vector to nominal baseline mode
        dists = np.linalg.norm(X_scaled - self.nominal_center, axis=1)

        # Normalized distance relative to baseline 95th percentile spread
        normalized_dists = dists / (self.baseline_distance_scale + 1e-9)
        raw_scores = (normalized_dists / 3.0) * 40.0

        # Adjust score according to cluster membership
        for i, c_id in enumerate(cluster_labels):
            c_int = int(c_id)
            if c_int == self.nominal_cluster_id:
                raw_scores[i] = min(raw_scores[i], self.threshold_warning - 5.0)
            elif c_int in self.cluster_health_profiles:
                c_status = self.cluster_health_profiles[c_int]["status"]
                if c_status == "WARNING":
                    raw_scores[i] = max(self.threshold_warning, min(raw_scores[i], self.threshold_critical - 2.0))
                elif c_status == "CRITICAL":
                    raw_scores[i] = max(self.threshold_critical + 2.0, raw_scores[i])
            elif c_int == -1:  # noise points
                raw_scores[i] = max(raw_scores[i], self.threshold_critical + 5.0)

        raw_scores = np.clip(raw_scores, 0.0, 100.0)

        # Temporal smoothing
        if smooth and len(raw_scores) > self.smoothing_window:
            smoothed_scores = (
                pd.Series(raw_scores)
                .rolling(window=self.smoothing_window, min_periods=1)
                .mean()
                .to_numpy()
            )
        else:
            smoothed_scores = raw_scores.copy()

        # Classify status based on smoothed score
        health_statuses = []
        for score in smoothed_scores:
            if score < self.threshold_warning:
                health_statuses.append("NORMAL")
            elif score < self.threshold_critical:
                health_statuses.append("WARNING")
            else:
                health_statuses.append("CRITICAL")

        return raw_scores, smoothed_scores, health_statuses

    def to_dict(self) -> Dict[str, Any]:
        """Serialize analyzer calibration state for persistence."""
        return {
            "baseline_method": self.baseline_method,
            "baseline_time_fraction": self.baseline_time_fraction,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical,
            "smoothing_window": self.smoothing_window,
            "nominal_cluster_id": self.nominal_cluster_id,
            "nominal_center": self.nominal_center.tolist() if self.nominal_center is not None else None,
            "baseline_distance_scale": float(self.baseline_distance_scale),
            "cluster_health_profiles": self.cluster_health_profiles,
            "is_calibrated": self.is_calibrated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthDegradationAnalyzer":
        """Reconstruct analyzer from serialized dictionary."""
        analyzer = cls(
            baseline_method=data.get("baseline_method", "earliest"),
            baseline_time_fraction=data.get("baseline_time_fraction", 0.25),
            threshold_warning=data.get("threshold_warning", 40.0),
            threshold_critical=data.get("threshold_critical", 75.0),
            smoothing_window=data.get("smoothing_window", 5),
        )
        analyzer.nominal_cluster_id = data.get("nominal_cluster_id")
        nom_center = data.get("nominal_center")
        analyzer.nominal_center = np.array(nom_center, dtype=np.float64) if nom_center is not None else None
        analyzer.baseline_distance_scale = float(data.get("baseline_distance_scale", 1.0))
        # Convert keys back to int
        profiles = data.get("cluster_health_profiles", {})
        analyzer.cluster_health_profiles = {int(k): v for k, v in profiles.items()}
        analyzer.is_calibrated = data.get("is_calibrated", False)
        return analyzer
