"""Mean-Shift clustering module for unsupervised operational state discovery."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


class MeanShiftEngine:
    """Manages bandwidth estimation, Mean-Shift model fitting, and cluster assignment."""

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        bandwidth_quantile: float = 0.2,
        bandwidth_n_samples: int = 500,
        bin_seeding: bool = True,
        cluster_all: bool = True,
    ):
        self.configured_bandwidth = bandwidth
        self.bandwidth_quantile = bandwidth_quantile
        self.bandwidth_n_samples = bandwidth_n_samples
        self.bin_seeding = bin_seeding
        self.cluster_all = cluster_all

        self.effective_bandwidth: Optional[float] = None
        self.model: Optional[MeanShift] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: int = 0

    def estimate_or_get_bandwidth(self, X: np.ndarray) -> float:
        """Estimate bandwidth using sklearn estimate_bandwidth or use configured override."""
        if self.configured_bandwidth is not None and self.configured_bandwidth > 0:
            print(f"[Clustering] Using user-configured bandwidth: {self.configured_bandwidth:.4f}")
            return float(self.configured_bandwidth)

        n_samples = len(X)
        sample_limit = min(self.bandwidth_n_samples, n_samples)

        try:
            bw = estimate_bandwidth(
                X,
                quantile=self.bandwidth_quantile,
                n_samples=sample_limit,
                random_state=42,
            )
            # Ensure estimated bandwidth is strictly positive
            if bw is None or bw <= 1e-6:
                raise ValueError("Estimated bandwidth was zero or negative.")
            print(f"[Clustering] Automatically estimated Mean-Shift bandwidth: {bw:.4f} (quantile={self.bandwidth_quantile})")
            return float(bw)
        except Exception as e:
            # Fallback estimate: fraction of mean pairwise Euclidean distance or standard deviation
            fallback_bw = float(np.mean(np.std(X, axis=0)) * 0.5)
            if fallback_bw <= 1e-6:
                fallback_bw = 1.0
            print(f"[Clustering] Bandwidth estimation notice ({e}). Using robust fallback bandwidth: {fallback_bw:.4f}")
            return fallback_bw

    def fit(self, X: np.ndarray) -> "MeanShiftEngine":
        """Fit MeanShift model on scaled feature matrix."""
        self.effective_bandwidth = self.estimate_or_get_bandwidth(X)

        self.model = MeanShift(
            bandwidth=self.effective_bandwidth,
            bin_seeding=self.bin_seeding,
            cluster_all=self.cluster_all,
            n_jobs=-1,
        )
        self.model.fit(X)

        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        self.n_clusters_ = len(self.cluster_centers_)

        print(f"[Clustering] Mean-Shift fitting completed. Discovered {self.n_clusters_} natural operating cluster(s).")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to the closest discovered cluster mode."""
        if self.model is None or self.cluster_centers_ is None:
            raise RuntimeError("MeanShiftEngine is not fitted yet.")

        # If model has scikit-learn predict method
        try:
            return self.model.predict(X)
        except Exception:
            # Fallback: Euclidean distance to cluster centers
            dists = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
            return np.argmin(dists, axis=1)

    def get_cluster_summary(self, labels: Optional[np.ndarray] = None) -> Dict[int, Dict[str, Any]]:
        """Compute cluster distributions and metrics."""
        lbls = self.labels_ if labels is None else labels
        if lbls is None:
            return {}

        total_pts = len(lbls)
        unique_labels, counts = np.unique(lbls, return_counts=True)

        summary = {}
        for c_id, count in zip(unique_labels, counts):
            summary[int(c_id)] = {
                "count": int(count),
                "percentage": round(float(count / total_pts * 100), 2),
                "is_noise": bool(c_id == -1),
            }
        return summary
