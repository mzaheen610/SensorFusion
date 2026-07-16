# Architecture

## Runtime Flow

The canonical implementation is `scripts/run_headless_slam.py`. It runs as a single Python process with separate reader threads for the IMU and LiDAR, then fuses their outputs in the main SLAM loop.

```text
RPLidar A2/M-series
  -> LidarReader thread
  -> filtered polar scan
  -> scan_to_xy()
  -> ICP against local submap
  -> EKF pose correction

ICM20948 IMU
  -> IMUReader thread
  -> acceleration and gyro samples
  -> tilt compensation and stationary checks
  -> EKF prediction and bias adaptation

EKF pose
  -> occupancy grid ray update
  -> periodic map image output
```

## Main Components

- `IMUReader` samples accelerometer and gyroscope data through Blinka and `adafruit_icm20x`.
- `LidarReader` reads RPLidar scans, applies the health-check monkey patch used in the prototype, and reconnects after scan errors.
- `PlanarEKF` estimates planar position, velocity, yaw, accelerometer bias, and gyro bias.
- `TiltEstimator` keeps roll and pitch estimates so acceleration can be leveled before planar prediction.
- `LocalSubmap` keeps recent transformed scans and downsamples them for ICP matching.
- `OccupancyGrid` applies log-odds updates with free-space ray carving and occupied endpoint marking.

## Canonical Implementation Choice

`scripts/run_headless_slam.py` was selected over the older `fusion_store_map.py` lineage because it includes the more mature estimator and map model: an 8-state EKF with bias terms, stationary adaptation, voxel downsampling, local submaps, stricter ICP gating, and free-space occupancy updates.

The archived files remain available under `experiments/archived/` for comparison and history, but new work should start from the canonical script.

## Known Technical Gaps

- No loop-closure or global pose-graph backend is implemented.
- No automated repeatability benchmark exists for drift or map quality.
- The script is still mostly monolithic; `src/sensorfusion/` is reserved for future module extraction.
- Hardware configuration is still mostly constants and CLI flags, not a full config file.
