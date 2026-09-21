"""Core pipeline orchestration for training, analysis, and prediction."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd

from .clustering import MeanShiftEngine
from .config import AppConfig, load_config
from .data_loader import PreprocessingPipeline, load_vibration_dataset
from .features import VibrationFeatureExtractor
from .health_analyzer import HealthDegradationAnalyzer
from .visualizer import (
    plot_cluster_distribution,
    plot_health_progression,
    plot_pca_clusters_2d,
    plot_sensor_trends,
)


def train_pipeline(config_path: str = "config.yaml", data_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute end-to-end unsupervised training of the Mean-Shift vibration monitoring system."""
    config: AppConfig = load_config(config_path)
    train_file = data_path or config.data.train_data_path

    print("=" * 78)
    print("  MACHINE HEALTH DEGRADATION MONITORING - MEAN-SHIFT TRAINING PIPELINE")
    print("=" * 78)
    print(f"Loading vibration dataset from: {train_file}")

    # 1. Load and validate data
    raw_df, timestamps = load_vibration_dataset(
        file_path=train_file,
        sensor_columns=config.data.sensor_columns,
        timestamp_column=config.data.timestamp_column,
        missing_strategy=config.preprocessing.missing_strategy,
    )
    print(f"Dataset loaded successfully: {len(raw_df)} measurements across {len(config.data.sensor_columns)} channels.")

    # 2. Extract vibration features
    print(f"\n[Feature Engineering] Extracting time & frequency domain vibration features...")
    extractor = VibrationFeatureExtractor(
        sensor_columns=config.data.sensor_columns,
        sampling_rate_hz=config.features.sampling_rate_hz,
        window_size=config.features.window_size,
        window_step=config.features.window_step,
        time_domain_features=config.features.time_domain_features,
        frequency_domain_features=config.features.frequency_domain_features,
    )
    features_df, window_timestamps = extractor.extract_windowed_features(raw_df, timestamps)
    print(f"Extracted {len(features_df)} feature windows with {features_df.shape[1]} descriptors each.")

    # 3. Fit preprocessing scaler
    print(f"\n[Preprocessing] Fitting {config.preprocessing.scaling_method.upper()} scaler...")
    preprocessor = PreprocessingPipeline(scaling_method=config.preprocessing.scaling_method)
    X_scaled = preprocessor.fit_transform(features_df)

    # 4. Mean-Shift Clustering
    print(f"\n[Clustering] Initializing Mean-Shift clustering...")
    clusterer = MeanShiftEngine(
        bandwidth=config.clustering.bandwidth,
        bandwidth_quantile=config.clustering.bandwidth_quantile,
        bandwidth_n_samples=config.clustering.bandwidth_n_samples,
        bin_seeding=config.clustering.bin_seeding,
        cluster_all=config.clustering.cluster_all,
    )
    clusterer.fit(X_scaled)

    # 5. Health degradation calibration
    print(f"\n[Health Analysis] Calibrating unsupervised baseline and degradation scale...")
    analyzer = HealthDegradationAnalyzer(
        baseline_method=config.health_analysis.baseline_method,
        baseline_time_fraction=config.health_analysis.baseline_time_fraction,
        threshold_warning=config.health_analysis.threshold_warning,
        threshold_critical=config.health_analysis.threshold_critical,
        smoothing_window=config.health_analysis.smoothing_window,
    )
    analyzer.calibrate(
        X_scaled=X_scaled,
        cluster_labels=clusterer.labels_,
        cluster_centers=clusterer.cluster_centers_,
    )

    # 6. Save models and artifacts
    models_dir = Path(config.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, models_dir / "scaler.joblib")
    joblib.dump(clusterer, models_dir / "mean_shift_model.joblib")
    joblib.dump(extractor, models_dir / "feature_pipeline.joblib")

    with open(models_dir / "health_baseline.json", "w", encoding="utf-8") as f:
        json.dump(analyzer.to_dict(), f, indent=2)

    # Save feature names list
    with open(models_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor.feature_names, f, indent=2)

    # 7. Print terminal summary
    print("\n" + "=" * 78)
    print("  TRAINING SUMMARY & CLUSTER CHARACTERIZATION")
    print("=" * 78)
    print(f"Total raw samples processed : {len(raw_df)}")
    print(f"Total feature windows       : {len(features_df)}")
    print(f"Discovered cluster modes    : {clusterer.n_clusters_}")
    print(f"Estimated bandwidth         : {clusterer.effective_bandwidth:.4f}")
    print(f"Nominal baseline cluster    : Cluster #{analyzer.nominal_cluster_id}")
    print(f"Baseline 95% distance scale : {analyzer.baseline_distance_scale:.4f}")

    print("\nCluster Health Breakdown:")
    print(f" {'Cluster':<9} | {'Windows':<8} | {'Dist to Baseline':<18} | {'Mean Score':<12} | {'Assigned Status'}")
    print("-" * 75)
    for c_id, profile in analyzer.cluster_health_profiles.items():
        print(
            f" #{c_id:<8} | {profile['sample_count']:<8} | "
            f"{profile['centroid_distance_to_baseline']:<18.4f} | "
            f"{profile['mean_degradation_score']:<12.2f} | "
            f"{profile['status']} - {profile['description']}"
        )

    print(f"\nArtifacts saved successfully to directory: {models_dir.resolve()}/")
    print("=" * 78)

    return {
        "n_samples": len(raw_df),
        "n_windows": len(features_df),
        "n_clusters": clusterer.n_clusters_,
        "bandwidth": clusterer.effective_bandwidth,
        "nominal_cluster": analyzer.nominal_cluster_id,
        "profiles": analyzer.cluster_health_profiles,
    }


def analyze_pipeline(config_path: str = "config.yaml") -> None:
    """Run comprehensive offline health degradation analysis and generate diagnostic visualizations."""
    config: AppConfig = load_config(config_path)
    models_dir = Path(config.paths.models_dir)

    # Ensure trained models exist
    scaler_path = models_dir / "scaler.joblib"
    model_path = models_dir / "mean_shift_model.joblib"
    extractor_path = models_dir / "feature_pipeline.joblib"
    baseline_path = models_dir / "health_baseline.json"

    if not all(p.is_file() for p in [scaler_path, model_path, extractor_path, baseline_path]):
        print("[Error] Required trained model artifacts not found in 'models/'. Run 'python main.py train' first.")
        return

    print("=" * 78)
    print("  MACHINE HEALTH DEGRADATION ANALYSIS & VISUALIZATION PIPELINE")
    print("=" * 78)

    # Load artifacts
    preprocessor: PreprocessingPipeline = joblib.load(scaler_path)
    clusterer: MeanShiftEngine = joblib.load(model_path)
    extractor: VibrationFeatureExtractor = joblib.load(extractor_path)

    with open(baseline_path, "r", encoding="utf-8") as f:
        analyzer = HealthDegradationAnalyzer.from_dict(json.load(f))

    # Load dataset
    raw_df, timestamps = load_vibration_dataset(
        file_path=config.data.train_data_path,
        sensor_columns=config.data.sensor_columns,
        timestamp_column=config.data.timestamp_column,
        missing_strategy=config.preprocessing.missing_strategy,
    )

    # Extract features & transform
    features_df, window_timestamps = extractor.extract_windowed_features(raw_df, timestamps)
    X_scaled = preprocessor.transform(features_df)

    # Compute degradation scores
    raw_scores, smoothed_scores, health_statuses = analyzer.compute_degradation_scores(
        X_scaled=X_scaled,
        cluster_labels=clusterer.labels_,
        smooth=True,
    )

    # Generate visualizations
    reports_dir = Path(config.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    dpi = config.visualization.dpi

    print("\nGenerating visual diagnostic reports...")
    p1 = plot_sensor_trends(
        raw_df,
        sensor_columns=config.data.sensor_columns,
        timestamp_col=config.data.timestamp_column,
        output_path=str(reports_dir / "sensor_trends.png"),
        dpi=dpi,
    )
    p2 = plot_cluster_distribution(
        analyzer.cluster_health_profiles,
        output_path=str(reports_dir / "cluster_distribution.png"),
        dpi=dpi,
    )
    p3 = plot_pca_clusters_2d(
        X_scaled=X_scaled,
        cluster_labels=clusterer.labels_,
        cluster_centers=clusterer.cluster_centers_,
        nominal_cluster_id=analyzer.nominal_cluster_id,
        output_path=str(reports_dir / "pca_cluster_2d.png"),
        dpi=dpi,
    )
    p4 = plot_health_progression(
        timestamps=window_timestamps,
        raw_scores=raw_scores,
        smoothed_scores=smoothed_scores,
        warning_threshold=analyzer.threshold_warning,
        critical_threshold=analyzer.threshold_critical,
        output_path=str(reports_dir / "health_progression.png"),
        dpi=dpi,
    )

    # Health progression statistics
    warning_indices = np.where(smoothed_scores >= analyzer.threshold_warning)[0]
    critical_indices = np.where(smoothed_scores >= analyzer.threshold_critical)[0]

    first_warning = window_timestamps.iloc[warning_indices[0]] if len(warning_indices) > 0 else "None detected"
    first_critical = window_timestamps.iloc[critical_indices[0]] if len(critical_indices) > 0 else "None detected"

    status_counts = pd.Series(health_statuses).value_counts().to_dict()

    print("\n" + "=" * 78)
    print("  TEMPORAL HEALTH DEGRADATION REPORT")
    print("=" * 78)
    print(f"Total Monitoring Windows  : {len(smoothed_scores)}")
    print(f"First Incipient Warning   : {first_warning}")
    print(f"First Critical Transition : {first_critical}")
    print("\nOperational State Windows:")
    print(f"  - NORMAL   : {status_counts.get('NORMAL', 0)} windows")
    print(f"  - WARNING  : {status_counts.get('WARNING', 0)} windows (Early defect indicators)")
    print(f"  - CRITICAL : {status_counts.get('CRITICAL', 0)} windows (Severe degradation)")
    print("\nGenerated Visualizations:")
    print(f"  1. {p1}")
    print(f"  2. {p2}")
    print(f"  3. {p3}")
    print(f"  4. {p4}")
    print("=" * 78)


def predict_pipeline(file_path: str, config_path: str = "config.yaml") -> Dict[str, Any]:
    """Perform real-time/batch health inference on new vibration measurements from the terminal."""
    config: AppConfig = load_config(config_path)
    models_dir = Path(config.paths.models_dir)

    scaler_path = models_dir / "scaler.joblib"
    model_path = models_dir / "mean_shift_model.joblib"
    extractor_path = models_dir / "feature_pipeline.joblib"
    baseline_path = models_dir / "health_baseline.json"

    for p in [scaler_path, model_path, extractor_path, baseline_path]:
        if not p.is_file():
            raise RuntimeError(
                f"Missing model artifact: '{p}'. Please run 'python main.py train' first."
            )

    # Load artifacts
    preprocessor: PreprocessingPipeline = joblib.load(scaler_path)
    clusterer: MeanShiftEngine = joblib.load(model_path)
    extractor: VibrationFeatureExtractor = joblib.load(extractor_path)

    with open(baseline_path, "r", encoding="utf-8") as f:
        analyzer = HealthDegradationAnalyzer.from_dict(json.load(f))

    # Load input data
    raw_df, timestamps = load_vibration_dataset(
        file_path=file_path,
        sensor_columns=config.data.sensor_columns,
        timestamp_column=config.data.timestamp_column,
        missing_strategy=config.preprocessing.missing_strategy,
    )

    # Extract features
    features_df, window_timestamps = extractor.extract_windowed_features(raw_df, timestamps)
    X_scaled = preprocessor.transform(features_df)

    # Assign clusters using Mean-Shift
    predicted_clusters = clusterer.predict(X_scaled)

    # Compute degradation score
    raw_scores, smoothed_scores, health_statuses = analyzer.compute_degradation_scores(
        X_scaled=X_scaled,
        cluster_labels=predicted_clusters,
        smooth=(len(features_df) > 3),
    )

    # Summary metrics
    avg_score = float(np.mean(smoothed_scores))
    max_score = float(np.max(smoothed_scores))

    # Primary machine condition based on worst/average trend
    if max_score >= analyzer.threshold_critical or health_statuses[-1] == "CRITICAL":
        overall_condition = "CRITICAL"
        rec = "CRITICAL FAILURE IMMINENT: High broadband vibration detected. Halt machine immediately for overhaul."
    elif max_score >= analyzer.threshold_warning or health_statuses[-1] == "WARNING":
        overall_condition = "WARNING"
        rec = "EARLY WEAR DETECTED: Incipient micro-impacts observed (elevated kurtosis/crest factor). Schedule inspection."
    else:
        overall_condition = "NORMAL"
        rec = "NORMAL HEALTH: Vibration amplitudes and spectral distribution within nominal tolerances."

    # Compute sensor statistics
    stats_table = []
    for col in config.data.sensor_columns:
        sig = raw_df[col].values
        rms = float(np.sqrt(np.mean(sig**2)))
        peak = float(np.max(np.abs(sig)))
        p2p = float(np.max(sig) - np.min(sig))
        cf = peak / (rms + 1e-9)
        from scipy import stats
        kurt = float(stats.kurtosis(sig, fisher=False)) if len(sig) > 3 else 3.0

        stats_table.append({
            "Sensor": col,
            "RMS (g)": round(rms, 4),
            "Peak (g)": round(peak, 4),
            "Peak-to-Peak": round(p2p, 4),
            "Crest Factor": round(cf, 2),
            "Kurtosis": round(kurt, 2),
        })

    # Terminal output
    print("\n" + "=" * 78)
    print("  MACHINE HEALTH INFERENCE & DEGRADATION MONITORING REPORT")
    print("=" * 78)
    print(f"Target Input File           : {file_path}")
    print(f"Total Vibration Samples     : {len(raw_df)}")
    print(f"Evaluated Feature Windows   : {len(features_df)}")
    print(f"Assigned Cluster Mode(s)    : {[int(c) for c in np.unique(predicted_clusters)]}")
    print(f"Mean Degradation Score      : {avg_score:.2f} / 100.0")
    print(f"Peak Degradation Score      : {max_score:.2f} / 100.0")
    print("-" * 78)

    status_tag = f"[{overall_condition}]"
    print(f"OVERALL MACHINE CONDITION   : {status_tag}")
    print(f"DIAGNOSTIC RECOMMENDATION   : {rec}")
    print("-" * 78)

    print("\nKey Sensor Vibration Statistics:")
    stat_df = pd.DataFrame(stats_table)
    print(stat_df.to_string(index=False))

    print("=" * 78 + "\n")

    return {
        "file": file_path,
        "n_samples": len(raw_df),
        "overall_condition": overall_condition,
        "avg_degradation_score": avg_score,
        "max_degradation_score": max_score,
        "predicted_clusters": [int(c) for c in np.unique(predicted_clusters)],
        "sensor_stats": stats_table,
    }
