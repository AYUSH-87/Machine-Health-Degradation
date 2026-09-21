# Machine Health Degradation Monitoring via Mean-Shift Clustering

An unsupervised machine learning system for industrial rotating machinery condition monitoring. The project applies **Mean-Shift clustering** on multi-sensor vibration time-series data to detect the earliest signs of mechanical degradation (such as incipient bearing spalling, shaft misalignment, or gear flaking) before catastrophic failure occurs.

Runs entirely from the terminal — no frontend, web UI, or heavy API servers required.

---

## 1. Problem Statement & Industrial Context

Mechanical assets (pumps, compressors, motors, gearboxes) exhibit distinct vibration signatures throughout their operational lifecycle:

1. **Nominal State**: Smooth harmonic motion, low vibration amplitudes, Gaussian amplitude distributions.
2. **Incipient Degradation (Early Warning)**: Periodic micro-impacts produced as rolling elements strike localized race defects. These transient shock waves sharply increase **Kurtosis** and **Crest Factor**, while overall RMS vibration remains deceptively low. This represents the **earliest window for predictive maintenance intervention**.
3. **Severe Degradation (Critical Failure)**: Continuous surface spalling, heavy broadband noise, severe shaft unbalance, and surging RMS amplitudes requiring immediate machine shutdown.

### Why Mean-Shift Clustering?
- **Unsupervised & Non-Parametric**: Traditional clustering algorithms (e.g., $k$-means) require specifying the number of clusters $k$ beforehand and assume spherical, equal-sized clusters. Mean-Shift discovers the natural modes (local maxima of the underlying probability density function) without prior assumptions.
- **Operating Regime Discovery**: Machines often operate across benign operating modes (varying speed, load levels). Mean-Shift groups these homogeneous conditions naturally.
- **Outlier & Novelty Isolation**: Degraded states form novel density clusters or drift into sparse, low-density regions away from healthy modes.

---

## 2. Unsupervised Health Degradation Methodology & Assumptions

Because Mean-Shift is an unsupervised algorithm, it produces cluster mode assignments ($0, 1, \dots, K-1$) without semantic labels like "healthy" or "failure". The pipeline bridges this gap using domain-grounded principles:

1. **Nominal Baseline Mode Assumption**:
   - Machinery is commissioned and initiated in a healthy operating state.
   - The cluster dominant during the initial monitoring window (or possessing the largest population support) is calibrated as the **Nominal Operational Mode** ($C_{\text{nominal}}$).
2. **Feature Distance Metric**:
   - For every monitoring window $i$, the normalized Euclidean distance $d_i = \| x_i - \mu_{\text{nominal}} \|_2$ from the feature vector to the nominal center is evaluated.
   - The 95th percentile distance of healthy baseline windows ($D_{\text{nominal}} = d_{95}$) defines the boundary of nominal operational variability.
3. **Continuous Health Degradation Score ($0 - 100$)**:
   $$S_i = \text{clip}\left(\frac{d_i}{2.5 \times D_{\text{nominal}}} \times 60.0, 0.0, 100.0\right)$$
   - Samples clustered in $C_{\text{nominal}}$ remain below the warning threshold ($< 40$).
   - Transient shock noise is filtered using a rolling Exponential Moving Average (EMA).
4. **Machine Health Status**:
   - **NORMAL** ($0 \le \text{Score} < 40$): Nominal operation; vibrations within acceptable tolerances.
   - **WARNING** ($40 \le \text{Score} < 75$): **Earliest signs of degradation** (micro-fault impacts, elevated kurtosis/crest factor). Maintenance inspection recommended.
   - **CRITICAL** ($\text{Score} \ge 75$): Severe mechanical deterioration; immediate shutdown recommended to prevent catastrophic failure.

---

## 3. Vibration Feature Engineering

For each sensor channel (e.g. tri-axial accelerometer: `accel_x`, `accel_y`, `accel_z`), the system extracts statistical time-domain and spectral frequency-domain indicators over sliding windows:

### Time-Domain Features
- **Root Mean Square (RMS)**: Total energy content ($\sqrt{\frac{1}{N}\sum x^2}$).
- **Standard Deviation ($\sigma$) & Variance ($\sigma^2$)**: Amplitude dispersion around mean.
- **Peak Value**: $\max(|x|)$.
- **Peak-to-Peak**: $\max(x) - \min(x)$.
- **Crest Factor**: $\frac{\text{Peak}}{\text{RMS}}$ (sensitive indicator of impulsive bearing impacts).
- **Kurtosis**: 4th standardized central moment ($\frac{\mu_4}{\sigma^4}$). Normal Gaussian signal equals 3.0; bearing race micro-impacts push kurtosis $> 4-8$.
- **Skewness**: 3rd standardized central moment ($\frac{\mu_3}{\sigma^3}$), measuring waveform asymmetry.
- **Shape Factor**: $\frac{\text{RMS}}{\text{Mean}(|x|)}$.
- **Impulse Factor**: $\frac{\text{Peak}}{\text{Mean}(|x|)}$.

### Frequency-Domain Features (FFT)
- **Dominant Frequency**: Frequency corresponding to the highest spectral peak amplitude.
- **Spectral Centroid**: Center of gravity of the frequency spectrum ($\frac{\sum f \cdot |X(f)|}{\sum |X(f)|}$).
- **Spectral Energy**: Sum of squared FFT spectral magnitudes ($\sum |X(f)|^2$).

---

## 4. Project Structure

```
.
├── config.yaml                     # Central YAML configuration
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Git ignore patterns
├── README.md                       # Comprehensive documentation
├── main.py                         # CLI entry point (train, analyze, predict, generate-data)
├── data/
│   ├── synthetic_vibration.csv     # Synthetic multi-sensor vibration dataset
│   └── samples/                    # Test snippets (test_normal.csv, test_warning.csv, test_critical.csv)
├── models/                         # Serialized models and calibration artifacts
│   ├── scaler.joblib               # Fitted preprocessing scaler (RobustScaler)
│   ├── mean_shift_model.joblib     # Fitted Mean-Shift clustering model
│   ├── feature_pipeline.joblib     # Vibration feature extractor configuration
│   ├── health_baseline.json        # Calibrated baseline cluster, distance scale, thresholds
│   └── feature_names.json          # Ordered list of feature names
├── reports/                        # Visual diagnostics generated by `analyze`
│   ├── sensor_trends.png           # Multi-sensor raw signals and rolling RMS
│   ├── cluster_distribution.png    # Cluster population and health statuses
│   ├── pca_cluster_2d.png          # 2D PCA projection of operating modes and centers
│   └── health_progression.png      # Degradation score trajectory across time
└── src/
    ├── __init__.py
    ├── config.py                   # Dataclass configuration loader
    ├── data_loader.py              # Data ingestion, cleaning, deduplication, imputation
    ├── features.py                 # Time-domain and spectral feature extraction
    ├── clustering.py               # Mean-Shift clustering with auto-bandwidth estimation
    ├── health_analyzer.py          # Baseline determination, degradation scoring, status logic
    ├── visualizer.py               # Headless Matplotlib publication-grade plotting
    ├── pipeline.py                 # Core orchestration routines
    └── synthetic_data.py           # Realistic multi-sensor vibration generator
```

---

## 5. Installation

Ensure Python 3.9+ is installed:

```bash
pip install -r requirements.txt
```

---

## 6. Terminal CLI Usage

### Step 1: Generate Demonstration Vibration Dataset
Generates realistic multi-channel vibration time-series simulating machine health deterioration across healthy, incipient wear, and severe failure regimes:

```bash
python main.py generate-data --samples 6000 --output data/synthetic_vibration.csv
```

*Note: Synthetic data is provided exclusively for pipeline demonstration and automated testing. For real-world use, supply your machinery's CSV data.*

### Step 2: Train Mean-Shift Model & Calibrate Health Baseline
Loads vibration time-series, extracts time/frequency features, scales data, automatically estimates Mean-Shift bandwidth, clusters operating modes, and calibrates the nominal baseline:

```bash
python main.py train --config config.yaml
```

### Step 3: Run Diagnostic Analysis & Generate Visualizations
Generates comprehensive diagnostic reports and publication-quality plots in `reports/`:

```bash
python main.py analyze --config config.yaml
```

Generated plots:
1. `reports/sensor_trends.png`: Multi-sensor raw amplitudes and rolling RMS envelopes.
2. `reports/cluster_distribution.png`: Sample distribution across discovered operating modes.
3. `reports/pca_cluster_2d.png`: High-dimensional feature space projected onto 2D PCA with mode centers.
4. `reports/health_progression.png`: Continuous health degradation timeline with NORMAL, WARNING, and CRITICAL threshold bands.

### Step 4: Predict Machine Health on New Measurements
Monitors new vibration data, assigns the operating cluster mode, calculates degradation score, and outputs a formatted diagnostic assessment in the terminal:

```bash
# Test normal healthy operation
python main.py predict --file data/samples/test_normal.csv

# Test incipient wear (earliest warning signs)
python main.py predict --file data/samples/test_warning.csv

# Test severe critical failure
python main.py predict --file data/samples/test_critical.csv
```

---

## 7. Configuration Reference (`config.yaml`)

Every parameter is fully configurable:

| Section | Parameter | Default | Description |
|---|---|---|---|
| `data` | `sensor_columns` | `[accel_x, accel_y, accel_z]` | List of accelerometer channel columns |
| `data` | `timestamp_column` | `timestamp` | Timestamp column name (or null) |
| `preprocessing` | `scaling_method` | `robust` | `robust` (median/IQR), `standard`, or `minmax` |
| `preprocessing` | `missing_strategy` | `forward_fill` | `forward_fill`, `median`, or `drop` |
| `features` | `window_size` | `200` | Raw samples per window chunk |
| `features` | `window_step` | `100` | Stride between consecutive windows |
| `features` | `sampling_rate_hz` | `1000.0` | Vibration sensor acquisition frequency |
| `clustering` | `bandwidth` | `null` | Explicit bandwidth or `null` for auto-estimation |
| `clustering` | `bandwidth_quantile` | `0.2` | Quantile used by `estimate_bandwidth` |
| `clustering` | `bin_seeding` | `true` | Accelerates Mean-Shift convergence |
| `health_analysis` | `baseline_method` | `earliest` | `earliest` (first fraction) or `largest` (mode size) |
| `health_analysis` | `threshold_warning`| `40.0` | Degradation score threshold for WARNING state |
| `health_analysis` | `threshold_critical`| `75.0` | Degradation score threshold for CRITICAL state |

---

## 8. License

MIT License. Designed for industrial condition monitoring and predictive maintenance research.
