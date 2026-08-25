# Sensor Fusion Tunable Parameters

This file documents tunable parameters in your fusion setup and what each one does.

## Which system is active right now

Current run path:
- test3.py -> lidar_IMU_fusion_full.py

So the most important parameters to tune are in `lidar_IMU_fusion_full.py`.

## 1) Hardware and sampling

| Parameter | Default | What it controls | Tuning notes |
|---|---:|---|---|
| `LIDAR_PORT` | `/dev/ttyUSB0` | LiDAR serial device | Change only if your LiDAR appears on a different port. |
| `LIDAR_BAUD` | `256000` | LiDAR serial baud rate | Must match sensor firmware. |
| `IMU_ADDR` | `0x69` | I2C address of ICM20948 | Use `0x68` on some boards. |
| `IMU_HZ` | `100` | IMU polling/update rate | Higher gives smoother integration but can increase noise load. |
| `DT_IMU` | `1.0 / IMU_HZ` | Nominal IMU time step | Usually derived, do not edit directly. |

## 2) Scan preprocessing

| Parameter | Default | What it controls | Tuning notes |
|---|---:|---|---|
| `MAX_RANGE_M` | `8.0` | Maximum accepted LiDAR range | Lower this in cluttered indoor scenes to reduce bad far returns. |
| `MIN_RANGE_M` | `0.15` | Minimum accepted LiDAR range | Raise if near-field noise is strong. |
| `MAX_SCAN_POINTS` | `420` | Scan downsample cap before ICP | Increase for richer geometry, decrease for lower CPU and more stability if noisy. |

## 3) ICP matching (critical for drift)

| Parameter | Default | What it controls | Tuning notes |
|---|---:|---|---|
| `ICP_ITER` | `20` | Max ICP iterations per scan | Increase if alignment is poor but CPU allows it. |
| `ICP_TOL` | `1e-3` | Convergence threshold | Lower for stricter convergence, higher for speed. |
| `ICP_MAX_CORR_M` | `0.80` | Max nearest-neighbor correspondence distance | Lower to reject wrong matches more aggressively. |
| `ICP_MAX_FITNESS` | `0.20` | Accept/reject threshold for ICP correction | Lower value = stricter correction acceptance, usually less drift when static. |
| `ICP_MIN_POINTS` | `70` | Minimum points to attempt ICP | Raise if sparse scans cause unstable corrections. |
| `ICP_MIN_PAIRS` | `40` | Minimum valid correspondence pairs | Raise to avoid weak updates, lower only if scans are very sparse. |
| `ICP_REJECT_RESET_COUNT` | `12` | Reinitialize keyframe after repeated ICP rejects | Lower for faster recovery, higher for conservative behavior. |

## 4) Keyframe policy

| Parameter | Default | What it controls | Tuning notes |
|---|---:|---|---|
| `KEYFRAME_DISTANCE_M` | `0.25` | Distance moved before keyframe refresh | Smaller updates keyframes more often; can reduce accumulated mismatch. |
| `KEYFRAME_YAW_RAD` | `10 deg` | Yaw change before keyframe refresh | Lower if rotational drift is dominant. |

## 5) Stationary detection and ZUPT (most important for static drift)

| Parameter | Default | What it controls | Tuning notes |
|---|---:|---|---|
| `ACC_NORM_TOL` | `0.40` | Allowed accel-norm error around 1g for stationary detection | Increase if detector misses stationary state; decrease if false stationary detections occur while moving slowly. |
| `GYRO_NORM_TOL` | `0.08` | Max gyro norm to be considered stationary | Lower for stricter stillness checks. |
| `STATIONARY_CONFIRM_SAMPLES` | `12` | Consecutive stationary confirmations needed | Increase to avoid flicker between moving/stationary states. |
| `ZUPT_STD` | `0.03` | Zero-velocity measurement noise for EKF | Lower value enforces stronger velocity clamping (less static drift). |

## 6) Bias calibration and adaptation

| Parameter | Default | What it controls | Tuning notes |
|---|---:|---|---|
| `CALIBRATION_SAMPLES` | `260` | Startup IMU bias calibration sample count | Increase for more stable bias estimate at startup. |
| `BIAS_ADAPT_ALPHA` | `0.01` | Rate of online bias adaptation while stationary | Lower for slower/safer adaptation, higher for faster bias tracking. |

## 7) EKF process-noise tuning

| Parameter | Default | What it controls | Tuning notes |
|---|---:|---|---|
| `ACC_NOISE_STD` | `0.45` | Acceleration process noise | Lower reduces velocity wander but can under-react to real motion. |
| `GYRO_NOISE_STD` | `0.05` | Gyro process noise | Lower reduces yaw drift but can over-trust gyro model. |
| `ACC_BIAS_RW_STD` | `0.01` | Accelerometer bias random-walk noise | Lower if bias should vary slowly. |
| `GYRO_BIAS_RW_STD` | `0.005` | Gyro bias random-walk noise | Lower this to reduce bias state wandering at rest. |

## 8) Occupancy map behavior

| Parameter | Default | What it controls | Tuning notes |
|---|---:|---|---|
| `MAP_CELL_M` | `0.05` | Grid resolution (meters per cell) | Lower value = finer map and more CPU/memory. |
| `MAP_CELLS` | `440` | Map width/height in cells | Larger map area costs memory. |
| `LOG_OCC` | `0.42` | Occupied log-odds increment | Higher marks hits more strongly. |
| `LOG_FREE` | `-0.16` | Free-space log-odds decrement | More negative clears free rays faster. |
| `LOG_CLIP_MIN` | `-5.0` | Min log-odds clamp | Keep as saturation bound. |
| `LOG_CLIP_MAX` | `5.0` | Max log-odds clamp | Keep as saturation bound. |

## 9) Output behavior

| Parameter | Default | What it controls | Tuning notes |
|---|---:|---|---|
| `SAVE_INTERVAL_S` | `2.0` | Plot save interval | Increase to reduce I/O load. |
| `OUTPUT_DIR` | `outputs` | Folder where plots are written | Change if you want a different output location. |

## 10) LiDAR stream recovery

| Parameter | Default | What it controls | Tuning notes |
|---|---:|---|---|
| `LIDAR_MAX_BUF_MEAS` | `12000` | Max internal RPLidar buffered measurements per scan iterator | Lower if memory/latency spikes; raise if buffer overruns happen under CPU load. |
| `LIDAR_SOFT_RECOVERY_MAX` | `4` | Max consecutive packet-desync soft recoveries before forcing hard reconnect | Lower to reconnect sooner when stream is unstable. |
| `LIDAR_SOFT_RECOVERY_SLEEP_S` | `0.03` | Delay after soft recovery before requesting a new iterator | Raise slightly if immediate re-read causes repeated descriptor errors. |
| `LIDAR_RECOVERY_WINDOW_S` | `12.0` | Sliding time window used to count recoverable stream errors | Shorter window reacts faster, longer window is more tolerant. |
| `LIDAR_MAX_RECOVERIES_PER_WINDOW` | `6` | Recoverable-error count within the window that triggers hard reconnect | Lower value is stricter and reconnects earlier. |
| `LIDAR_RECONNECT_DELAY_S` | `0.20` | Delay before retrying full LiDAR reconnect | Increase if USB/serial stack needs more settle time after disconnect. |

## Fast tuning recipe for static drift

If the setup is static but trajectory still moves:

1. Reduce `ZUPT_STD` (example: `0.03 -> 0.02` or `0.015`).
2. Reduce `GYRO_BIAS_RW_STD` (example: `0.005 -> 0.003`).
3. Tighten ICP acceptance by reducing `ICP_MAX_FITNESS` (example: `0.20 -> 0.14`).
4. Tighten stationary detection by reducing `GYRO_NORM_TOL` slightly (example: `0.08 -> 0.06`) if false static detections are not a problem.
5. Increase `CALIBRATION_SAMPLES` (example: `260 -> 400`) and keep the rig perfectly still during startup.

## Legacy scripts (for reference)

These scripts still contain older tunables:
- `fusion_store_map.py`
- `test2.py`

Important legacy-only parameters include:
- `ICP_THRESH`
- `ZUPT_ACC_TOL`
- `ZUPT_GYRO_TOL`
- `ZUPT_VEL_NOISE`
- `ICP_MAX_FITNESS_FOR_REF`

Use those only if you continue running those files directly.
