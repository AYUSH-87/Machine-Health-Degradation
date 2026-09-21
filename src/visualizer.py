"""Visualization module for machine health degradation and Mean-Shift clustering.

Generates publication-quality plots saved directly to disk (headless mode):
1. Multi-sensor trends over time
2. Cluster distribution and operational state breakdown
3. PCA 2D projection of natural operating modes and cluster centers
4. Health degradation progression timeline with threshold boundaries
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for terminal environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def plot_sensor_trends(
    df: pd.DataFrame,
    sensor_columns: List[str],
    timestamp_col: Optional[str] = "timestamp",
    output_path: str = "reports/sensor_trends.png",
    dpi: int = 200,
) -> str:
    """Generate time-series trend plot of multi-sensor vibration signals."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(sensor_columns), 1, figsize=(12, 3.2 * len(sensor_columns)), sharex=True)
    if len(sensor_columns) == 1:
        axes = [axes]

    # Create x-axis
    if timestamp_col and timestamp_col in df.columns:
        x_vals = pd.to_datetime(df[timestamp_col], errors="coerce")
        if x_vals.isna().all():
            x_vals = np.arange(len(df))
            x_label = "Sample Index"
        else:
            x_label = "Timestamp"
    else:
        x_vals = np.arange(len(df))
        x_label = "Sample Index"

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (col, ax) in enumerate(zip(sensor_columns, axes)):
        color = colors[idx % len(colors)]
        ax.plot(x_vals, df[col], color=color, linewidth=0.8, alpha=0.85, label=f"{col} Raw Amplitude")
        # Overlay moving RMS envelope
        rolling_rms = df[col].rolling(window=100, min_periods=1).apply(lambda s: np.sqrt(np.mean(s**2)))
        ax.plot(x_vals, rolling_rms, color="black", linewidth=1.4, linestyle="--", label="Rolling RMS (Window=100)")

        ax.set_ylabel(f"{col} (g)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[-1].set_xlabel(x_label, fontsize=11, fontweight="bold")
    fig.suptitle("Industrial Machine Multi-Sensor Vibration Trends Over Time", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[Visualizer] Saved sensor trends plot -> {out}")
    return str(out)


def plot_cluster_distribution(
    cluster_profiles: Dict[int, Dict],
    output_path: str = "reports/cluster_distribution.png",
    dpi: int = 200,
) -> str:
    """Generate bar chart of cluster population and health classifications."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    clusters = list(cluster_profiles.keys())
    counts = [cluster_profiles[c]["sample_count"] for c in clusters]
    statuses = [cluster_profiles[c]["status"] for c in clusters]

    status_color_map = {
        "NORMAL": "#2ecc71",
        "WARNING": "#f39c12",
        "CRITICAL": "#e74c3c",
    }
    bar_colors = [status_color_map.get(s, "#95a5a6") for s in statuses]
    labels = [f"Cluster #{c}\n({statuses[i]})" for i, c in enumerate(clusters)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts, color=bar_colors, edgecolor="black", linewidth=1.1, width=0.55)

    # Annotate counts on top of bars
    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max(counts) * 0.015,
            f"{count} windows",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylabel("Number of Feature Windows", fontsize=11, fontweight="bold")
    ax.set_title("Discovered Mean-Shift Operating Condition Cluster Distribution", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Legend for health status colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="NORMAL State"),
        Patch(facecolor="#f39c12", edgecolor="black", label="WARNING State (Early Incipient Signs)"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="CRITICAL State (Severe Degradation)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.95)

    plt.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[Visualizer] Saved cluster distribution plot -> {out}")
    return str(out)


def plot_pca_clusters_2d(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    nominal_cluster_id: int,
    output_path: str = "reports/pca_cluster_2d.png",
    dpi: int = 200,
) -> str:
    """Project high-dimensional vibration feature space onto 2D via PCA and plot clusters."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(cluster_centers)

    explained_var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 7))

    unique_labels = np.unique(cluster_labels)
    try:
        cmap = matplotlib.colormaps["tab10"]
    except Exception:
        cmap = plt.get_cmap("tab10")

    for idx, c_id in enumerate(unique_labels):
        mask = cluster_labels == c_id
        pts = X_pca[mask]
        is_nom = (c_id == nominal_cluster_id)
        label_str = f"Cluster #{c_id}" + (" (Nominal Baseline)" if is_nom else "")

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=28,
            alpha=0.65,
            color=cmap(idx),
            label=label_str,
            edgecolors="none",
        )

    # Plot cluster centers
    for idx, (c_id, center_pt) in enumerate(zip(range(len(cluster_centers)), centers_pca)):
        is_nom = (c_id == nominal_cluster_id)
        marker = "P" if is_nom else "X"
        ax.scatter(
            center_pt[0],
            center_pt[1],
            s=220,
            marker=marker,
            color="black",
            edgecolors="gold" if is_nom else "white",
            linewidth=2.0,
            zorder=10,
        )
        ax.annotate(
            f"Mode #{c_id}",
            (center_pt[0], center_pt[1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontweight="bold",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="grey"),
        )

    ax.set_xlabel(f"Principal Component 1 ({explained_var[0]:.1f}% variance)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Principal Component 2 ({explained_var[1]:.1f}% variance)", fontsize=11, fontweight="bold")
    ax.set_title("Mean-Shift Operating Condition Modes in 2D Feature Space (PCA)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[Visualizer] Saved 2D PCA cluster visualization -> {out}")
    return str(out)


def plot_health_progression(
    timestamps: pd.Series,
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    warning_threshold: float = 40.0,
    critical_threshold: float = 75.0,
    output_path: str = "reports/health_progression.png",
    dpi: int = 200,
) -> str:
    """Plot temporal progression of machine health degradation score across operation timeline."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Time axis
    t_vals = pd.to_datetime(timestamps, errors="coerce")
    if t_vals.isna().all():
        x_axis = np.arange(len(raw_scores))
        x_label = "Monitoring Window Index"
    else:
        x_axis = t_vals
        x_label = "Timestamp"

    # Background color bands for operational health zones
    max_score = max(100.0, float(np.max(smoothed_scores) + 10.0))

    ax.axhspan(0, warning_threshold, color="#2ecc71", alpha=0.15, label="NORMAL Operating Zone")
    ax.axhspan(warning_threshold, critical_threshold, color="#f39c12", alpha=0.18, label="WARNING Zone (Incipient Wear)")
    ax.axhspan(critical_threshold, max_score, color="#e74c3c", alpha=0.20, label="CRITICAL Zone (Imminent Failure)")

    # Plot raw and smoothed degradation scores
    ax.plot(x_axis, raw_scores, color="#7f8c8d", alpha=0.45, linewidth=1.0, label="Raw Window Degradation Score")
    ax.plot(x_axis, smoothed_scores, color="#2c3e50", linewidth=2.4, label="Smoothed Health Degradation Trend")

    # Threshold horizontal reference lines
    ax.axhline(warning_threshold, color="#d35400", linestyle="--", linewidth=1.6, label=f"Warning Threshold ({warning_threshold})")
    ax.axhline(critical_threshold, color="#c0392b", linestyle="-.", linewidth=1.8, label=f"Critical Threshold ({critical_threshold})")

    ax.set_ylim(-2.0, max_score)
    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Health Degradation Score (0 = Nominal, 100 = Critical)", fontsize=11, fontweight="bold")
    ax.set_title("Industrial Machine Health Degradation Timeline & Progression", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.95, ncol=2)

    plt.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[Visualizer] Saved health degradation progression plot -> {out}")
    return str(out)
