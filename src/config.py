"""Configuration management for Machine Health Degradation Monitoring.

Loads, validates, and provides structured access to settings defined in config.yaml.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class DataConfig:
    train_data_path: str = "data/synthetic_vibration.csv"
    timestamp_column: Optional[str] = "timestamp"
    sensor_columns: List[str] = field(default_factory=lambda: ["accel_x", "accel_y", "accel_z"])


@dataclass
class PreprocessingConfig:
    scaling_method: str = "robust"  # "robust", "standard", "minmax"
    missing_strategy: str = "forward_fill"  # "forward_fill", "median", "drop"


@dataclass
class FeaturesConfig:
    sampling_rate_hz: float = 1000.0
    window_size: int = 200
    window_step: int = 100
    time_domain_features: List[str] = field(
        default_factory=lambda: [
            "rms", "std", "variance", "peak", "peak_to_peak",
            "crest_factor", "kurtosis", "skewness", "shape_factor", "impulse_factor"
        ]
    )
    frequency_domain_features: List[str] = field(
        default_factory=lambda: ["dominant_frequency", "spectral_centroid", "spectral_energy"]
    )


@dataclass
class ClusteringConfig:
    bandwidth: Optional[float] = None
    bandwidth_quantile: float = 0.2
    bandwidth_n_samples: int = 500
    bin_seeding: bool = True
    cluster_all: bool = True


@dataclass
class HealthConfig:
    baseline_method: str = "earliest"  # "earliest" or "largest"
    baseline_time_fraction: float = 0.25
    threshold_warning: float = 40.0
    threshold_critical: float = 75.0
    smoothing_window: int = 5


@dataclass
class PathsConfig:
    models_dir: str = "models"
    reports_dir: str = "reports"
    data_dir: str = "data"


@dataclass
class VisualizationConfig:
    dpi: int = 200
    style: str = "seaborn-v0_8-whitegrid"
    figure_format: str = "png"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    health_analysis: HealthConfig = field(default_factory=HealthConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration from a YAML file into typed AppConfig."""
    path = Path(config_path)
    if not path.is_file():
        # Fall back to default config if file is absent
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    config = AppConfig(
        data=DataConfig(**data.get("data", {})),
        preprocessing=PreprocessingConfig(**data.get("preprocessing", {})),
        features=FeaturesConfig(**data.get("features", {})),
        clustering=ClusteringConfig(**data.get("clustering", {})),
        health_analysis=HealthConfig(**data.get("health_analysis", {})),
        paths=PathsConfig(**data.get("paths", {})),
        visualization=VisualizationConfig(**data.get("visualization", {})),
    )

    # Ensure output directories exist
    Path(config.paths.models_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.reports_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.data_dir).mkdir(parents=True, exist_ok=True)

    return config
