# Multimodal Sensor Fusion: System Knowledge Base & Architectural Reference

> **Purpose of this document**: This document serves as the single source of truth for the system architecture, mathematical foundations, hyperparameter thresholds, historical failures, and lessons learned. **Before modifying any threshold or architectural component, consult this document to avoid re-introducing previously resolved failure modes.**

---

## 1. System Overview & Hardware Architecture

The platform is an autonomous multimodal sensor fusion system combining **LiDAR**, **IMU**, and **Monocular Visual Odometry** on resource-constrained embedded hardware (Raspberry Pi).

```
                      +-----------------------------+
                      |   Adafruit BNO055 (I2C)     |
                      | 9-DOF IMU (accel/gyro/grav) |
                      +--------------+--------------+
                                     | ~33 Hz (I2C)
                                     v
+------------------------+    +---------------+    +------------------------+
| RPLidar A1/A2 (Serial) |    |  imu_thread   |    | PiCamera2 (CSI bus)    |
| 14.5 Hz 360-deg scans  |    | (Main process)|    | 2.5 Hz keyframes       |
+-----------+------------+    +-------+-------+    +-----------+------------+
            |                         |                        |
            v (multiprocessing Queue) | (Lock-free buffers)    v
+-----------+------------+            |            +-----------+------------+
| lidar_acquisition_proc |            |            |     camera_thread      |
| (Dedicated worker)     |            |            |  Photometric Residuals |
+-----------+------------+            |            +-----------+------------+
            |                         |                        |
            v                         v                        v
+-----------+-------------------------+------------------------+------------+
|                             ESIKF State Estimator                         |
|   State x = [R, p, v, bg, ba, g]^T in SO(3) x R^15 (18 error dimensions)  |
|                                                                           |
|   - IMU Prediction: Forward numerical integration (30-33 Hz)              |
|   - LiDAR Update: Point-to-Plane Iterated EKF (0.3 - 0.7s per scan)       |
|   - Camera Update: Semi-direct Photometric Residual Alignment (2.5 Hz)    |
|   - Global Map: 0.5m Hash Voxel Grid with SVD Planar/Line Features        |
+-------------------------------------+-------------------------------------+
                                      |
                                      v (TCP Socket :5000)
                         +------------+------------+
                         | Live Laptop Web Streamer|
                         | (Open3D / Three.js GUI) |
                         +-------------------------+
```

### Coordinate System (ROS REP-103 Standard)
- **$+X$**: Forward (longitudinal axis of the robot / cart).
- **$+Y$**: Left (lateral axis of the robot / cart).
- **$+Z$**: Up (vertical axis perpendicular to the ground).
- **Yaw ($\psi$)**: Rotation around $+Z$. Clockwise turn (turn right) produces negative angular velocity ($\omega_z < 0$). Counter-clockwise turn produces positive angular velocity ($\omega_z > 0$).
- **LiDAR native raw reading**: RPLidar reports raw clockwise angles from its optical index. `backward.py` converts this to standard right-handed counter-clockwise radians via: `(360.0 - raw_angle) % 360.0`.

---

## 2. Concurrency & Multi-Threading Model

| Thread / Process | Execution Rate | Primary Responsibilities | Critical State Shared |
| :--- | :--- | :--- | :--- |
| **`imu_thread`** | $30\text{--}33\text{ Hz}$ | Reads BNO055 via I2C, runs forward ESIKF prediction, logs historical `imu_measurement_buffer` and `imu_state_buffer`. | Writes `filter.state`, `filter.P`. Reads `filter.is_static`. |
| **`lidar_acquisition_proc`** | $14.5\text{ Hz}$ | Runs as a separate OS process (`multiprocessing.Process`). Drains serial buffer without blocking fusion loop. Pushes latest complete 360° scan to single-slot `scan_queue`. | Communicates via `multiprocessing.Queue`. |
| **`lidar_thread`** | $1.5\text{--}3.0\text{ Hz}$ ($\sim 0.3\text{--}0.7\text{ s}$ per scan) | Retrieves scan from queue, performs backward motion de-skewing (`backprop`), queries voxel map, computes SVD plane normals, executes IESKF measurement update, computes multi-modal ZUPT, reconciles velocity. | Locks `state_lock`, merges $\Delta p, \Delta R$, updates `map`. |
| **`camera_thread`** | $\sim 2.5\text{ Hz}$ | Captures frames, projects LiDAR map voxels onto image plane, extracts pixel patches, evaluates photometric residuals, updates pose. | Locks `state_lock`, merges camera $\Delta p, \Delta R$. |
| **`stream_thread`** | $\sim 2.0\text{ Hz}$ | Serializes map points and latest trajectory coordinates into JSON over TCP port 5000 for remote browser visualizer. | Reads under `buffer_lock`. |

---

## 3. Thresholds & Hyperparameters Directory

Every threshold in this table was tuned to solve a specific real-world failure on hardware. **Do not change any value without reviewing its "Failure If Tweaked" condition.**

| Parameter | Location | Current Value | Engineering Purpose / Rationale | Failure Mode If Tweaked |
| :--- | :--- | :--- | :--- | :--- |
| **`ZUPT_CONSECUTIVE_REQUIRED`** | `update.py:38` | `2` | Requires 2 consecutive resting scans ($\sim 0.15\text{ s}$) before asserting static lock. | If `1`: False-positive stationary lock during slow crawl.<br>If `>3`: High latency before catching drift at a stop. |
| **`gyro_max` threshold** | `update.py:262` | `0.12 rad/s` ($\approx 6.9^\circ/\text{s}$) | Separates true platform turning from sensor noise and chassis vibration. | If `<0.08`: Never enters static mode on a running robot.<br>If `>0.20`: Slow physical turns mistakenly classified as stationary. |
| **`accel_var` threshold** | `update.py:262` | `0.40` | Accommodates mechanical vibration of the spinning LiDAR motor ($14.5\text{ Hz}$). | If `<0.30`: Motor vibration causes `accel_var` to hover at $0.35$, permanently blocking static mode (seen in Log `11_39_32`).<br>If `>0.80`: Real hand pushes classified as static. |
| **`sim` threshold (Scan Similarity)** | `update.py:273` | `35.0 mm` | Median bin distance between consecutive scans. Resting baseline is $1\text{--}15\text{ mm}$; real motion produces $\ge 40\text{ mm}$. | If `<25.0`: False dynamic triggers on sensor noise.<br>If `>60.0`: Slow creep/crawl undetected by LiDAR stillness check. |
| **`MAX_STATIONARY_DRIFT`** | `update.py:318` | `0.03 m` ($3\text{ cm}$) | Clamps cumulative position deviation from the resting anchor while stationary. | If omitted: Self-exciting circular orbits drift $3.5\text{ m}$ (Log `12_33_07`).<br>If `<0.01`: Fails to absorb natural sensor noise floor. |
| **Static Heading Lock** | `update.py:328` | Locked to `static_anchor_R` | Freezes yaw rotation while `is_platform_static = True`. | If unlocked: 30 Hz gyro integration accumulates heading drift that drives orbital circles. |
| **Bias Adaptation Dwell** | `update.py:292` | `static_count >= 10` ($\approx 0.7\text{ s}$) | Enforces sustained standstill before fine-tuning accelerometer and gyroscope biases. | If `<5`: Adapts bias during transient stops or slow turns, learning legitimate motion as bias. |
| **`mean_gyro_norm` for Bias** | `update.py:298` | `0.08 rad/s` ($\approx 4.6^\circ/\text{s}$) | Strict guard ensuring platform is truly still before modifying $b_g$. | If `>0.15`: Slow manual turns contaminate gyro bias. |
| **`search_radius`** | `forward.py:221` | `1` (static) / `2` (dynamic) | Voxel search window ($1 = \pm 0.5\text{ m}, 2 = \pm 1.0\text{ m}$) for nearest neighbor associations. | If static is `2`: Matches distant points across room.<br>If dynamic is `1`: Loses tracking during fast hand push ($> 0.5\text{ m/s}$). |
| **`gate` (Residual Gate)** | `forward.py:350` | `max(min_gate, 4*sigma)` with `min_gate = 0.80` (r=2) / `0.40` (r=1) | Dynamic innovation residual outlier gate scaled by state uncertainty and search radius. | If rigidly capped (<0.25m): Longitudinal end-wall beams during fast forward hand pushes are rejected as outliers, collapsing Fisher Information along X and causing severe lateral slide (Issue 6).<br>If unbounded (>1.5m): Matches perpendicular walls across room during turns. |
| **`MAX_BODY_LATERAL_DISP`** | `forward.py:473` | `0.03 m` ($3\text{ cm}$) | Non-Holonomic Constraint (NHC): Clamps single-scan lateral displacement in the robot body frame ($dx_{body}[1]$) to wheel slip limits ($\le 3\text{ cm}$). | If omitted: Solver solves yaw residuals against side walls by sliding the robot sideways into walls during turns or longitudinal strokes. |
| **`v_obs_body[1]` (Velocity NHC)** | `update.py:360` | `0.0 m/s` | Non-Holonomic Constraint on velocity reconciliation: Projects observed velocity into body frame and forces lateral body velocity to zero. | If omitted: Discontinuous lateral ICP noise synthesizes lateral velocity, causing open-loop sideways IMU acceleration into walls. |
| **`turn_settled` Latch** | `update.py:418` | `0.5 s` dwell after turn | Prevents committing new LiDAR map points during or immediately following active rotation until angular rate has settled for $\ge 0.5\text{ s}$. | If omitted: Ghost walls are permanently committed during rapid turns while orientation error is settling. |
| **`is_turning` threshold** | `update.py:344, 416` | `gyro_max > 0.20 rad/s` ($\approx 11.5^\circ/\text{s}$) | Suppresses velocity synthesis and map point ingestion during active angular turns. | If omitted: Discontinuous rotation shifts synthesize linear velocity, causing runaway trajectories. |
| **`disp` jump threshold** | `update.py:345` | `0.12 m` per scan or $> 1.2\text{ m/s}$ | Disallows discontinuous ICP jumps from being converted to vehicle velocity. | If `>0.25`: Jumps are reconciled as velocity, sustaining open-loop flying trajectories. |
| **`alpha_v` (Reconciliation)** | `update.py:361` | `0.7` | Pulls live IMU velocity towards true LiDAR-committed velocity. | If `0.0`: IMU velocity drifts open loop.<br>If `1.0`: Single noisy scan update causes velocity jitter. |
| **`max_position_correction`** | `forward.py:449` | `clip(3*sigma_p, 0.15, 0.40)` (static) / `0.25..0.50` (dynamic) | Upper bound on single-scan IESKF translation step. | Prevents filter divergence if scan associations are corrupted. |
| **`max_rotation_correction`** | `forward.py:452` | `deg2rad(25.0)` ($\approx 0.436\text{ rad}$) | Upper bound on single-scan IESKF rotation step. | Prevents sudden $90^\circ$ orientation flips on ambiguous geometry. |

---

## 4. Chronological Evolution & Known Issues Log

### Issue 1: High Motor Vibration DC Rectification
- **Manifestation (Log `run_2026_10_13_18_02_23.log`)**: Robot sitting completely stationary on a table drifted across the visualizer room at $0.4\text{ m/s}$.
- **Root Cause**: The spinning RPLidar motor ($14.5\text{ Hz}$) introduces high-frequency mechanical oscillation. The raw accelerometer reading is sinusoidal with zero mean. However, computing Euclidean norm $\|\mathbf{a}\|$ on noisy 3-axis readings acts as a mathematical rectifier ($\mathbb{E}[\|\mathbf{n}\|] > 0$), converting zero-mean vibration into a constant $+0.35\text{ m/s}^2$ fictitious forward acceleration.
- **Solution Applied**: In `src/sensorfusion/forward.py`, implemented a horizontal 2D vector EMA filter (`_static_accel_ema_vec`) *before* taking the norm. True zero-mean motor vibrations cancel out symmetrically to $< 0.05\text{ m/s}^2$.

---

### Issue 2: Accel and Gyro Bias Runaway
- **Manifestation (Log `run_2026_10_13_15_42_43.log`)**: Velocity grew unbounded to $> 2.0\text{ m/s}$; filter drifted into infinity.
- **Root Cause**:
  1. LiDAR ICP and Camera VO updates attempted to estimate accelerometer bias ($b_a$) and gyro bias ($b_g$). Because position-only observations cannot observe accelerometer bias without high-frequency acceleration excitation, the solver pushed unobservable state errors into the bias states.
  2. ZUPT bias adaptation was running continuously without checking if the robot was actually resting.
- **Solution Applied**:
  1. In `src/sensorfusion/forward.py` and `src/sensorfusion/camera/photometry.py`, permanently zeroed out Kalman gain blocks for unobservable dimensions ($K_{accel\_bias} = 0, K_{gyro\_bias} = 0$ in LiDAR/camera).
  2. Confined bias adaptation strictly to confirmed stationary intervals with dwell time `static_count >= 10`, `mean_gyro < 0.08 rad/s`, and `sim < 25.0 mm`.

---

### Issue 3: Backward Trajectory Collapse at Corridor Dead-Ends
- **Manifestation (Log `run_2026_10_14_12_08_59.log`)**: The robot was pushed forward along a straight corridor towards an end wall. When approaching the wall, the estimated trajectory suddenly reversed and collapsed backwards by $1.8\text{ m}$.
- **Root Cause**: In `src/sensorfusion/map.py`, line 89 had a legacy early-exit:
  ```python
  if dx == -2 and len(all_candidates) > 0:
      stop_search = True
  ```
  This created an asymmetric KD-tree search: for negative coordinates (behind the robot), it stopped searching after 1 voxel, while for positive coordinates (in front), it searched all 5 voxels. When reaching the end wall, associations in front dropped while rear associations dominated, pulling the position backwards.
- **Solution Applied**: Removed `stop_search` completely; gathered candidates symmetrically across all directions $[-r, +r]$; added dynamic distance cutoff `max_correspondence_distance = (radius + 0.5) * voxel_size`.

---

### Issue 4: Stationary Circular Runaway Orbits
- **Manifestation (Log `run_2026_10_14_12_33_07.log`)**: The platform was placed completely stationary on the floor for 60 seconds. The estimated trajectory began orbiting in continuous $360^\circ$ circles with a diameter of $3.5\text{ meters}$.
- **Root Cause**:
  1. High-frequency gyro integration ($30\text{ Hz}$) accumulated tiny sub-degree random walk drift.
  2. At each LiDAR update, the small orientation error caused LiDAR point-to-plane matching to displace position along a circle tangent.
  3. Velocity reconciliation saw successive displaced positions and synthesized linear velocity ($0.15\text{ m/s}$).
  4. IMU prediction integrated this velocity, creating a self-sustaining perpetual orbital loop.
- **Solution Applied (Commit `dbb75f2`)**:
  1. Created a stationary resting anchor (`static_anchor_pos`, `static_anchor_R`) in `update.py`. Total displacement while static is hard-clamped to $\le 0.03\text{ m}$ ($3\text{ cm}$).
  2. Heading rotation integration in `forward.py` is frozen when `is_static = True`.
  3. Pre-initialized `filter.is_static = True` and `static_count = 2` before launching worker threads.
- **Verification (Log `run_2026_10_14_13_24_39.log`)**: Complete elimination of orbits. Stationary drift reduced from $3.57\text{ m}$ to $< 1.3\text{ cm}$ over 10-second resting periods.

---

### Issue 5: Sideways 1.2-Meter Position Jump During 90-Degree Right Turn
- **Manifestation (Log `run_2026_10_14_13_37_18.log`)**: The robot was pushed straight forward along a corridor ($X \approx 0 \to 1.0\text{ m}$), then turned $90^\circ$ to the right in place. The trajectory abruptly shot sideways across the corridor, jumping $1.24\text{ m}$ into the right wall.
- **Root Cause**:
  1. **Jacobian Degeneracy**: When turning to face a flat perpendicular wall ($n = [0, 1, 0]$), the rotation Jacobian $H_{rot} = -n^T R [p]_\times$ vanishes to zero. The solver cannot minimize residuals by rotating; it can only minimize residuals by translating laterally ($H_{pos} = [0, 1, 0]$).
  2. **Wide Residual Gate**: With `search_radius = 2` ($1.0\text{ m}$) and `min_gate = 1.20\text{ m}`, beams sweeping towards the right wall matched with residuals of $0.5\text{--}0.8\text{ m}$. Instead of rejecting them as outliers, the solver translated the robot $0.24\text{ m}$ sideways per scan.
  3. **Unlatched Turn Gating**: When manual turning speed dipped to $0.09\text{ rad/s}$ ($< 0.20$), `is_turning` evaluated to `False`. Velocity reconciliation synthesized $v_y = -0.186\text{ m/s}$, accelerating the jump.
  4. **Map Contamination**: `map.add_points` ingested 317 points at the jumped position ($Y = -0.93\text{ m}$ to $-1.18\text{ m}$), permanently locking the robot into the wrong pose.
- **Solution Roadmap**: Tighten residual outlier gate; enforce non-holonomic constraint ($dx_{body, y} \le 0.03\text{ m}, v_{body, y} = 0$); latch map point additions until rotation settles for $\ge 0.5\text{ s}$.

---

### Issue 6: Forward/Backward Straight-Line Hand-Push Degeneracy & Lateral Drift under Rigid 0.25m Gate Cap
- **Manifestation (Logs `run_2026_10_14_15_50_20.log`, `run_2026_10_14_15_54_59.log`, `run_2026_10_14_16_00_22.log`)**:
  The robot was pulled forward and backward along a straight corridor over $\pm 1.0\text{ m}$. Instead of tracking a straight forward/backward line along $X$, the estimated state in `run_16_00_22.log` jumped $1.10\text{ m}$ laterally into the side wall ($Y \in [-0.984, 0.116]$) while $X$ only tracked $0.62\text{ m}$. In `run_15_54_59.log`, 1988 beams were rejected by the gate.
- **Root Cause**:
  1. **Longitudinal Beam Starvation**: To prevent Issue 5, the residual gate was temporarily capped rigidly to `np.clip(k_sigma, 0.15, 0.25)`. During manual forward hand pulls ($> 0.4\text{ m/s}$), longitudinal displacement between scans caused end-wall residuals of $0.25\text{--}0.40\text{ m}$. The $0.25\text{ m}$ cap rejected all forward and rear beams as outliers (up to 13 beams rejected per scan).
  2. **Corridor Information Degeneracy**: With longitudinal beams rejected, only parallel side-wall beams ($n \approx [0, 1, 0]^T$) remained in the optimization. The measurement Jacobian row for side walls has $H_{trans} = [0, 1, 0]$ ($H_x = 0$). The Fisher Information matrix $H^T H$ in the longitudinal $X$ direction completely collapsed to zero.
  3. **Yaw-to-Lateral Lever-Arm Crosstalk**: Any slight hand wobble ($2^\circ\text{--}5^\circ$) during the pull tilted side-wall beams, producing residuals $r_i \approx x_i \sin\delta\psi \approx 0.10\text{--}0.20\text{ m}$. Because these were $< 0.25\text{ m}$, they passed the gate. With zero rank along $X$, the solver minimized these residuals entirely by translating the robot sideways in $Y$ by $-0.17\text{ m}$ to $-0.23\text{ m}$ per scan.
  4. **Unconstrained Velocity Reconciliation**: Velocity reconciliation computed $v_y = -0.27\text{ m/s}$ in the world frame and injected it into the filter, causing IMU integration to slide the robot open-loop into the wall.
- **Solution Applied**:
  1. **Dynamic Residual Gate**: Restored dynamic gate `gate = max(min_gate, 4 * sigma)` with `min_gate = 0.80 m` for `search_radius >= 2` and `0.40 m` for `search_radius == 1`, allowing longitudinal forward strokes and blackout recovery to associate without being discarded.
  2. **Non-Holonomic Constraint (NHC) on IESKF Step**: In `forward.py`, transformed $dx[3:6]$ into body frame ($dx_{body} = R^T dx[3:6]$) and strictly clamped lateral body displacement to wheel slip limits: $dx_{body}[1] = \text{clip}(dx_{body}[1], -0.03, 0.03)\text{ m}$, and $dx_{body}[2] = 0.0$. Transformed back: $dx[3:6] = R \cdot dx_{body}$.
  3. **Non-Holonomic Constraint (NHC) on Velocity Reconciliation**: In `update.py`, projected observed velocity onto body frame ($v_{obs\_body} = R^T v_{obs}$), forced lateral and vertical body velocity to zero ($v_{obs\_body}[1] = 0.0, v_{obs\_body}[2] = 0.0$), and transformed back.
  4. **Turn Settling Latch**: Required angular velocity to remain settled ($\le 0.20\text{ rad/s}$) for $\ge 0.5\text{ s}$ after any turn before admitting new points into `map.add_points`.
- **Verification**: All 80 unit test scenarios passed (`OK`), including $0.6\text{ m}$ forward stroke recovery, $0.35\text{ m}$ initial offset recovery, $2.5\text{ s}$ blackout recovery, and stationary drift benchmarks.

---

## 5. Architectural Comparison with SOTA (FAST-LIVO2 / LIO-SAM)

| Feature | SOTA (FAST-LIO2 / FAST-LIVO2 / LIO-SAM) | Our Implementation | Impact & Required Alignment |
| :--- | :--- | :--- | :--- |
| **Outlier Rejection** | Point-to-plane residual strictly gated by $|pd2| < 0.111 \sqrt{\|p\|}$ ($\le 11\text{ cm}$ at $1\text{ m}$). | Hardcoded gate up to $1.20\text{ m}$. | **Critical**: Tighten residual outlier rejection so rotated beams cannot match cross-room walls. |
| **Degeneracy Handling** | Eigenvalue decomposition of $H^T H$. Degenerate directions zeroed out from LiDAR update and left to IMU. | Unconstrained inversion of $(H^T R^{-1} H + P^{-1})$. | **High**: In degenerate geometries (facing flat walls), damp lateral translation during turns. |
| **Kinematic Constraints** | Non-holonomic constraint (NHC): $v_{body, y} \approx 0, v_{body, z} \approx 0$ on wheeled ground vehicles. | Free 3D unconstrained translation in world frame. | **High**: Ground robot cannot slide sideways; clamp body-lateral translation step. |
| **Map Management** | Points added only after low-residual convergence; never during angular acceleration. | Added whenever `disp > 0.10m` and instantaneous `gyro < 0.20`. | **High**: Require a multi-scan turn settling latch before allowing new map points. |
| **Pipeline Latency** | Synchronous: IMU propagates to scan end time $t_k$; IESKF updates state at $t_k$; next IMU propagation starts from $t_k$. | Asynchronous: LiDAR thread takes $0.5\text{ s}$; delta $\Delta p$ computed in old heading is merged into current live pose. | **Medium**: Keep scan processing under $0.2\text{ s}$ via point downsampling ($N \le 40$). |

---

## 6. Rules of Engagement for Future Code Changes

1. **Never adjust a threshold in isolation**: A threshold that fixes dynamic tracking (e.g. widening gates) can destroy stationary stability. Always check the impact on both regimes.
2. **Mandatory Test Verification**:
   Before committing any change, run the complete 80-scenario unit test suite:
   ```bash
   python3 -m unittest discover tests
   ```
   All tests must pass (`OK`).
3. **Log Replay Verification**:
   Verify modifications against the reference log benchmarks:
   - `run_2026_10_14_12_33_07.log`: Verifies stationary circular runaway suppression.
   - `run_2026_10_14_13_24_39.log`: Verifies stationary dwell drift $< 2\text{ cm}$ and hand-push stroke tracking.
   - `run_2026_10_14_13_37_18.log`: Verifies $90^\circ$ right turn stability without lateral jumping.
4. **Updating this Knowledge Base**:
   Whenever a threshold is adjusted, a bug is fixed, or a new edge case is discovered, update this document immediately with the commit hash, root cause, and verified outcome.
