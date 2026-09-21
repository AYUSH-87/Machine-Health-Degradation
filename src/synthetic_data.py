"""Synthetic multi-sensor vibration dataset generator.

Simulates industrial rotating machinery vibration across three mechanical degradation phases:
1. Stage 1 (Healthy/Nominal): Baseline shaft harmonics + low Gaussian noise (low RMS, kurtosis ~3).
2. Stage 2 (Incipient Degradation): Periodic micro-impacts (early bearing race flaking),
   spiking kurtosis and crest factor with slight RMS elevation (earliest signs of failure).
3. Stage 3 (Severe Degradation): Heavy broadband unbalance, surging RMS and peak amplitudes,
   chaotic transient excursions.

NOTE: Synthetic data is solely for pipeline verification, testing, and demonstration.
In real deployments, feed calibrated tri-axial or multi-channel accelerometer recordings.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def generate_synthetic_vibration(
    num_samples: int = 6000,
    sampling_rate_hz: float = 1000.0,
    sensor_columns: Optional[List[str]] = None,
    output_path: Optional[str] = "data/synthetic_vibration.csv",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-sensor vibration time-series simulating machine health degradation.

    Parameters
    ----------
    num_samples : int
        Total number of time-series samples to generate.
    sampling_rate_hz : float
        Simulated sampling rate in Hz (e.g. 1000 Hz = 1 ms per point).
    sensor_columns : list of str, optional
        Names of sensor columns (defaults to ['accel_x', 'accel_y', 'accel_z']).
    output_path : str, optional
        Path where the generated CSV should be saved.
    random_seed : int
        Seed for deterministic reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing timestamps, sensor channels, and simulated operational phase.
    """
    if sensor_columns is None:
        sensor_columns = ["accel_x", "accel_y", "accel_z"]

    np.random.seed(random_seed)

    time_step = 1.0 / sampling_rate_hz
    t = np.arange(num_samples) * time_step

    # Fundamental rotation frequency (e.g., 25 Hz = 1500 RPM motor)
    f_rot = 25.0
    # Bearing defect characteristic frequency (e.g., 120 Hz ball pass outer race)
    f_defect = 120.0

    # Operational stages:
    # 0 -> 40%: Healthy
    # 40% -> 75%: Incipient fault (early warning: high kurtosis/crest factor)
    # 75% -> 100%: Severe failure (critical: high RMS, large broadband vibration)
    idx_stage1 = int(num_samples * 0.40)
    idx_stage2 = int(num_samples * 0.75)

    signals: dict = {col: np.zeros(num_samples) for col in sensor_columns}
    operational_phase: List[str] = []

    # Axis-specific phase offsets and sensitivity factors
    axis_modifiers = {
        "accel_x": {"phase": 0.0, "scale": 1.0, "impact_bias": 1.2},
        "accel_y": {"phase": np.pi / 2, "scale": 0.85, "impact_bias": 0.9},
        "accel_z": {"phase": np.pi / 4, "scale": 0.70, "impact_bias": 1.5},
    }

    # Pulse train for bearing impact simulation (damped exponentials)
    impact_interval = int(sampling_rate_hz / f_defect)  # samples between defect hits
    damped_envelope = np.exp(-np.linspace(0, 5, impact_interval))

    for i in range(num_samples):
        # Determine operational phase
        if i < idx_stage1:
            phase = "NORMAL_HEALTHY"
            amp_rot = 0.35
            noise_sigma = 0.05
            impact_amp = 0.0
        elif i < idx_stage2:
            phase = "INCIPIENT_WARNING"
            # Linear ramp up of micro-fault impulses
            progress = (i - idx_stage1) / (idx_stage2 - idx_stage1)
            amp_rot = 0.35 + 0.15 * progress
            noise_sigma = 0.07 + 0.03 * progress
            impact_amp = 0.60 * (0.3 + 0.7 * progress)
        else:
            phase = "SEVERE_CRITICAL"
            progress = (i - idx_stage2) / (num_samples - idx_stage2)
            amp_rot = 0.60 + 0.90 * progress
            noise_sigma = 0.15 + 0.25 * progress
            impact_amp = 1.20 + 0.80 * progress

        operational_phase.append(phase)

        # Impact impulse contribution
        impact_idx = i % impact_interval
        impact_val = impact_amp * damped_envelope[impact_idx] * np.sin(2 * np.pi * 350 * t[i])

        for col in sensor_columns:
            mod = axis_modifiers.get(col, {"phase": 0.0, "scale": 1.0, "impact_bias": 1.0})
            phi = mod["phase"]
            scale = mod["scale"]
            impact_factor = mod["impact_bias"]

            # 1X shaft rotation + 2X harmonic + impact pulse + Gaussian noise
            val = (
                amp_rot * np.sin(2 * np.pi * f_rot * t[i] + phi)
                + 0.15 * amp_rot * np.sin(2 * np.pi * (2 * f_rot) * t[i] + phi)
                + impact_val * impact_factor
                + np.random.normal(0, noise_sigma)
            ) * scale

            signals[col][i] = round(float(val), 5)

    # Generate synthetic timestamps starting from a realistic baseline
    start_time = datetime(2026, 3, 15, 8, 0, 0)
    timestamps = [start_time + timedelta(seconds=float(s)) for s in t]

    df = pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] for ts in timestamps],
        **signals,
        "simulated_ground_truth": operational_phase,
    })

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[DataGen] Generated {len(df)} synthetic vibration samples -> {out}")
        print(f"[DataGen] Columns: {list(df.columns)}")
        print(f"[DataGen] Simulated ground truth distribution:\n{df['simulated_ground_truth'].value_counts().to_string()}")

    return df


def generate_sample_monitoring_chunks(
    source_df: pd.DataFrame,
    output_dir: str = "data/samples",
    window_size: int = 300,
) -> Tuple[str, str, str]:
    """Generate three isolated sample CSV files representing NORMAL, WARNING, and CRITICAL states.

    Useful for validating `python main.py predict --file ...` on distinct machine health conditions.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_df = source_df[source_df["simulated_ground_truth"] == "NORMAL_HEALTHY"].iloc[:window_size]
    warning_df = source_df[source_df["simulated_ground_truth"] == "INCIPIENT_WARNING"].iloc[
        len(source_df[source_df["simulated_ground_truth"] == "INCIPIENT_WARNING"]) // 2 :
    ].iloc[:window_size]
    critical_df = source_df[source_df["simulated_ground_truth"] == "SEVERE_CRITICAL"].iloc[-window_size:]

    p_norm = str(out_dir / "test_normal.csv")
    p_warn = str(out_dir / "test_warning.csv")
    p_crit = str(out_dir / "test_critical.csv")

    # Save without ground truth column to test production inference behavior
    drop_cols = [c for c in ["simulated_ground_truth"] if c in source_df.columns]
    normal_df.drop(columns=drop_cols).to_csv(p_norm, index=False)
    warning_df.drop(columns=drop_cols).to_csv(p_warn, index=False)
    critical_df.drop(columns=drop_cols).to_csv(p_crit, index=False)

    print(f"[DataGen] Created test monitoring samples:")
    print(f"  - Normal:   {p_norm} ({len(normal_df)} rows)")
    print(f"  - Warning:  {p_warn} ({len(warning_df)} rows)")
    print(f"  - Critical: {p_crit} ({len(critical_df)} rows)")

    return p_norm, p_warn, p_crit


if __name__ == "__main__":
    df = generate_synthetic_vibration()
    generate_sample_monitoring_chunks(df)
