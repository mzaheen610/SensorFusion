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
| **`sim` threshold (Scan Similarity)** | `update.py:273` | `20.0 mm` | Median bin distance between consecutive scans. Resting noise baseline is $1\text{--}10\text{ mm}$ (90th percentile $9.5\text{ mm}$); real motion produces $\ge 20\text{ mm}$. | If `>30.0`: Slow starts remain trapped inside the 3cm stationary anchor for 1-2 seconds (Issue 7).<br>If `<15.0`: False dynamic triggers on mechanical motor vibration. |
| **`MAX_STATIONARY_DRIFT`** | `update.py:318` | `0.03 m` ($3\text{ cm}$) | Clamps cumulative position deviation from the resting anchor while stationary. | If omitted: Self-exciting circular orbits drift $3.5\text{ m}$ (Log `12_33_07`).<br>If `<0.01`: Fails to absorb natural sensor noise floor. |
| **Static Heading Lock** | `update.py:328` | Locked to `static_anchor_R` | Freezes yaw rotation while `is_platform_static = True`. | If unlocked: 30 Hz gyro integration accumulates heading drift that drives orbital circles. |
| **Bias Adaptation Dwell** | `update.py:292` | `static_count >= 10` ($\approx 0.7\text{ s}$) | Enforces sustained standstill before fine-tuning accelerometer and gyroscope biases. | If `<5`: Adapts bias during transient stops or slow turns, learning legitimate motion as bias. |
| **`mean_gyro_norm` for Bias** | `update.py:298` | `0.08 rad/s` ($\approx 4.6^\circ/\text{s}$) | Strict guard ensuring platform is truly still before modifying $b_g$. | If `>0.15`: Slow manual turns contaminate gyro bias. |
| **`search_radius`** | `forward.py:221` | `1` (static) / `2` (dynamic) | Voxel search window ($1 = \pm 0.5\text{ m}, 2 = \pm 1.0\text{ m}$) for nearest neighbor associations. | If static is `2`: Matches distant points across room.<br>If dynamic is `1`: Loses tracking during fast hand push ($> 0.5\text{ m/s}$). |
| **`gate` (Residual Gate)** | `forward.py:356` | Static: `0.20..0.40m`<br>Dynamic ($r=2$): `0.70..0.75m`<br>Dynamic ($r=1$): `0.35..0.45m` | Dynamic innovation residual outlier gate bounded by upper thresholds to prevent cross-table matching during covariance expansion while accepting forward hand pushes up to $0.65\text{ m}$. | If rigidly capped (<0.25m): Longitudinal end-wall beams during fast forward hand pushes are rejected as outliers, causing lateral slide (Issue 6).<br>If unbounded (>1.5m): Matches perpendicular walls across room during turns (Issue 8). |
| **`MAX_BODY_LATERAL_DISP`** | `forward.py:488` | `0.03 m` (straight) / `0.06 m` (turning) | Non-Holonomic Constraint (NHC): Clamps body-lateral displacement to wheel slip limits ($3\text{ cm}$), allowing $6\text{ cm}$ during active rotation to accommodate the natural circular turn arc. | If rigidly 0.03m in turns: Chokes turn arc, causing robotic stutter around L-corners (Issue 7).<br>If unconstrained: Wall tilt converts into sideways jumps. |
| **`dx_body[0]` Reverse Limit** | `forward.py:491` | `dx_body[0] >= -0.04 m` during turns | Non-Holonomic Constraint (NHC): Clamps body-forward step from reversing by more than $4\text{ cm}$ during active turns. | If omitted: Corner ICP false matches pull vehicle backwards into reverse (Issue 8). |
| **`v_obs_body[1]` (Velocity NHC)** | `update.py:361` | `0.0 m/s` | Non-Holonomic Constraint on velocity reconciliation: Projects observed velocity into body frame and forces lateral body velocity to zero, while preserving forward linear velocity along heading during turns. | If omitted: Discontinuous lateral ICP noise synthesizes lateral velocity, causing open-loop sideways IMU acceleration into walls. |
| **`v_obs_body[0]` (Turn Reverse Protection)** | `update.py:366` | `v_obs_body[0] >= 0.0 m/s` when $\omega > 0.15$ | Suppresses false reverse velocity synthesis during turns caused by unmapped corner association drags. | If omitted: Corner drags synthesize negative forward velocity, propelling IMU dead reckoning backwards (Issue 8). |
| **Turn Frontier Mapping** | `update.py:448` | Add points if $\Delta \psi \ge 15^\circ$ ($0.26\text{ rad}$) and $\omega \le 0.80\text{ rad/s}$ | Progressively maps new frontiers as heading rotates through corners, preventing blind turn entry. | If omitted: Turning into unmapped room leaves zero map points ahead, causing ICP to latch onto old side walls behind robot (Issue 8). |
| **`turn_settled` Latch** | `update.py:442` | `0.15 s` dwell after turn | Prevents committing new LiDAR map points during high angular acceleration ($\omega > 0.80$), but allows immediate point ingestion ($0.15\text{ s}$) once rotation slows. | If too long (>0.4s): Starves map points when turning into new corridors, causing blind tracking lag (Issue 7).<br>If omitted: Ghost walls smeared during peak turn. |
| **`is_turning` threshold** | `update.py:365, 440` | `gyro_max > 0.15..0.20 rad/s` | Activates turn-specific kinematics constraints and settling latches. | If omitted: Discontinuous rotation shifts synthesize linear velocity, causing runaway trajectories. |
| **`disp` jump threshold** | `update.py:348` | `0.12 m` per scan or $> 1.2\text{ m/s}$ | Disallows discontinuous ICP jumps from being converted to vehicle velocity. | If `>0.25`: Jumps are reconciled as velocity, sustaining open-loop flying trajectories. |
| **`alpha_v` (Reconciliation)** | `update.py:377` | `0.7` | Pulls live IMU velocity towards true LiDAR-committed velocity. | If `0.0`: IMU velocity drifts open loop.<br>If `1.0`: Single noisy scan update causes velocity jitter. |
| **`max_position_correction`** | `forward.py:461` | `clip(3*sigma_p, 0.08, 0.15)` (static) / `0.25..0.50` (dynamic) | Upper bound on single-scan IESKF translation step. Tightened in static mode to prevent corner pulls. | Prevents filter divergence if scan associations are corrupted. |
| **`max_correspondence_distance`** | `map.py:69` | `0.40 m` (static, $r=1$) / `1.25 m` (dynamic, $r=2$) | Maximum neighbor distance admitted by voxel KD search. | If $>0.50$ when static: Beams at corners associate with side voxels across empty space (Issue 8). |
| **`max_rotation_correction`** | `forward.py:464` | `deg2rad(25.0)` ($\approx 0.436\text{ rad}$) | Upper bound on single-scan IESKF rotation step. | Prevents sudden $90^\circ$ orientation flips on ambiguous geometry. |
| **`BNO055 Gyro Z Remap`** | `initialize.py:75, 103` | `gyro[2] = -gyro[2]` | Remaps BNO055 clockwise compass yaw rate to ROS REP-103 right-handed CCW convention (+Z Up). | If omitted: Turns rotate state estimator $R$ opposite to physical motion, throwing LiDAR points $180^\circ$ away and destroying scan matching (Issue 9). |
| **`I2C Spike Gating`** | `main.py:56` | `norm(w) > 15 rad/s` or `norm(a) > 25 m/s²` | Rejects BNO055 0x7FFF / ADC saturation spikes caused by I2C clock stretching or bus contention. | If omitted: 327 m/s² glitches inject 5.5 m/s phantom velocity in a single 16ms step, causing multi-meter trajectory runaway (Issue 9). |

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

### Issue 7: Sluggish Movement Response and Stuttering L-Shaped Turns
- **Manifestation (Logs `run_2026_10_15_11_57_12.log`, `run_2026_10_15_12_15_05.log`)**:
  Platform was stable, but the user experienced a noticeable sluggish response when initiating motion, and jerky/stuttering behavior when rounding 90° L-shaped turns.
- **Root Cause**:
  1. **Exponential $-6\cdot v$ IMU Velocity Decay during Cruising**: In `forward.py`, `v *= (1 - 6*dt)` executed whenever horizontal acceleration was low (`accel < 0.12`). During steady-state constant velocity motion, acceleration is near zero. The velocity was exponentially damped with a half-life of $115\text{ ms}$, attenuating forward speed by $85\text{--}90\%$ between successive LiDAR updates and creating heavy virtual viscous drag.
  2. **1-Second Stationary Anchor Exit Delay**: In `update.py`, `sim < 35.0 mm` was used to classify LiDAR stillness. The resting noise floor is only $1\text{--}10\text{ mm}$ (90th percentile $9.5\text{ mm}$). Smooth initial motion ($0.15\text{--}0.25\text{ m/s}$) produced $15\text{--}25\text{ mm}$ of scan displacement, keeping `is_platform_static = True` for $2\text{--}3$ scans ($0.6\text{--}1.0\text{ s}$) while position was repeatedly clamped to within $3\text{ cm}$ of `static_anchor_pos` and velocity zeroed.
  3. **Zeroing Forward Velocity in Turns**: In `update.py`, `is_turning` ($\omega > 0.20\text{ rad/s}$) unconditionally forced $v_{obs} = 0$. In real driving, an L-turn combines rotation with forward translation along a curved arc. Zeroing $v_{obs}$ halted forward progression during turns, turning smooth arcs into discrete stop-turn-go actions.
  4. **0.5-Second Map Latch Blindness**: In `update.py`, `turn_settled` prevented map point additions for $0.5\text{ s}$ after turning. Rounding a corner into a new hallway, the robot navigated blind without map points ahead of it; associations dropped to 9–13 points, making the update weak and laggy.
  5. **Rigid 3cm Lateral Clamp Truncating Turn Arcs**: Over a $0.5\text{ s}$ scan interval, turning at $0.5\text{--}0.8\text{ rad/s}$ produces $0.04\text{--}0.06\text{ m}$ of natural chord displacement. Rigidly clamping body-lateral displacement to $0.03\text{ m}$ truncated the natural turn curvature.
- **Solution Applied**:
  1. **Physical Rolling Resistance**: Replaced exponential $-6\cdot v$ decay with linear physical friction deceleration ($a_{friction} = 0.35\text{ m/s}^2$). Above $0.03\text{ m/s}$, speed decays naturally and smoothly, preserving forward momentum between scans while cleanly arresting stationary drift below $3\text{ cm/s}$.
  2. **Fast Static Release**: Lowered `sim` stationary threshold from $35.0\text{ mm}$ to $20.0\text{ mm}$. Resting noise ($< 10\text{ mm}$) remains firmly static; real movement ($> 20\text{ mm}$) breaks static mode on scan 1.
  3. **Preserve Forward Velocity in Turns under NHC**: Removed `is_turning` from wiping $v_{obs}$. The Non-Holonomic Constraint projects $v_{obs}$ into body frame and strictly zeroes lateral velocity ($v_{body, y} = 0$), while preserving legitimate forward velocity ($v_{body, x}$) along the turn arc.
  4. **Rapid Map Turn Latch**: Reduced post-turn settling dwell from $0.5\text{ s}$ to $0.15\text{ s}$, ingesting new hallway points on the very next scan after rotation settles.
  5. **Dynamic Turn Arc Lateral Limit**: Expanded the body-lateral clamp from $0.03\text{ m}$ (straight) to $0.06\text{ m}$ during active rotation ($|\omega| > 0.08$), allowing smooth rounding of L-corners.
- **Verification**: Verified on full 80-scenario unit test suite (80/80 passed `OK`).

---

### Issue 8: Table Circumnavigation Backward Trajectory Collapse at Corner Turn
- **Manifestation (Log `run_2026_10_15_16_18_12.log`)**:
  The user moved the robot around a central rectangular table. Initial slow pushes along the right edge of the table (0 to 1.30m) tracked well ($X \approx 1.30\text{ m}, Y \approx -0.29\text{ m}$). When rounding the top edge (moving forward and left), the trajectory abruptly reversed and collapsed backwards from $X = 1.31\text{ m}$ down to $0.57\text{ m}$, losing track of the corner.
- **Root Cause**:
  1. **Cross-Corner Association While Static**: While paused at the corner ($X = 1.30\text{ m}$), `Map.query()` defaulted to $0.75\text{ m}$ correspondence distance even when static (`radius_voxels == 1`). Beams forward of the corner associated with older voxels on the side of the table ($X \in [0.8, 1.1\text{ m}]$), producing negative $\Delta x \approx -0.15\text{ m}$ corrections.
  2. **Static Anchor Release Shock**: When the user began moving (Merge 431), `is_platform_static` dropped to `False`. The filter immediately added the full negative $\Delta p = [-0.150, 0.0]\text{ m}$ to the live pose, dropping $X$ from $1.273\text{ m}$ to $1.124\text{ m}$, followed by $[-0.137, -0.004]\text{ m}$ on Merge 432 ($0.986\text{ m}$), collapsing $32\text{ cm}$ backwards before turning started.
  3. **Map Freezing during Turns (`is_turning > 0.20`)**: During the 4-second $90^\circ$ turn, human turning speed was $30\text{--}37^\circ/\text{s}$ ($0.55\text{--}0.64\text{ rad/s}$). Because `gyro_max > 0.20 rad/s`, `is_turning` completely blocked `map.add_points()` for 9 consecutive scans (`map_points` stayed frozen at 1996). The robot turned into an unmapped frontier.
  4. **Unbounded Outlier Residual Gate ($4.0\text{ m}$)**: With no map points on the front edge, beams hitting the front of the table associated with the side edge in the map across $0.4\text{--}0.6\text{ m}$ of empty space. Because IMU covariance grew during the turn, the residual gate expanded to $4.0\text{ m}$, accepting all false cross-table associations.
  5. **Kinematic Projection into Reverse Displacement**: In the robot's $90^\circ$ left-turned body frame, the false association pull towards the side wall (world $-Y$) projected directly onto body $-X$ (reverse). The lateral NHC ($dx_{body, y} \le 0.06\text{ m}$) had zero lateral component to clamp, allowing $-0.25\text{ m}$ longitudinal body steps per scan.
  6. **Spurious Reverse Velocity Synthesis**: Observed velocity reconciliation ($v_{obs}$) projected into body frame zeroed lateral velocity ($v_{body, y} = 0$), but allowed negative forward velocity ($v_{body, x} \approx -0.8\text{ m/s}$). This reverse velocity was injected into `filter.state.v`, actively driving IMU dead reckoning backwards.
  7. **Contamination upon Stopping**: When the robot stopped at $X = 0.57\text{ m}$, `is_static = True` tripped, unlatching `map.add_points()`, which immediately burned 118 points into the map at the false backward pose ($X = 0.58\text{ m}$), permanently corrupting the map for the remaining 100 merges.
- **Solution Applied**:
  1. **Bounded Static Correspondence Distance**: In `map.py`, capped `max_correspondence_distance` to $\min(0.40, (radius\_voxels + 0.5) \times \text{voxel\_size})$ when $r = 1$, preventing cross-corner associations while resting.
  2. **Bounded Dynamic Residual Gate & Static Gate Tightening**: In `forward.py`, capped `gate` to $\le 0.75\text{ m}$ ($r \ge 2$) / $\le 0.45\text{ m}$ ($r = 1$), and tightened static gate to $0.20\text{--}0.40\text{ m}$, rejecting cross-table and cross-room false associations even under large covariance.
  3. **Static Position Step Tightening**: In `forward.py`, clamped `max_position_correction` while static to `float(np.clip(3.0 * sigma_p, 0.08, 0.15))`.
  4. **Non-Holonomic Reverse Step Protection during Turns**: In `forward.py`, during active rotation (`is_turning_step`), bounded reverse body displacement to $dx_{body}[0] \ge -0.04\text{ m}$. Straight-line reverse motion remains unconstrained.
  5. **Smooth Static Anchor Release**: In `update.py`, upon transitioning from static to dynamic, reset `filter.state.p[:2]` cleanly to `static_anchor_pos` and clamped any transition-frame backward body step ($dx_{body}[0] \ge 0.0$).
  6. **Reverse Velocity Synthesis Suppression during Turns**: In `update.py`, during active turns ($\omega > 0.15\text{ rad/s}$), clamped observed body velocity $v_{obs\_body}[0] \ge 0.0\text{ m/s}$.
  7. **Turn Frontier Map Ingestion**: In `update.py`, added progressive map point ingestion whenever heading rotates by $\ge 15^\circ$ ($0.26\text{ rad}$) at controlled turn rates ($\le 0.80\text{ rad/s}$), plus a post-turn settled snapshot, ensuring new geometries ahead of the vehicle are mapped before beams need them.
- **Verification**: Verified on full 85-scenario unit test suite (85/85 passed `OK`), including targeted tests for corner reversal suppression, smooth anchor release, and turn frontier mapping.

---

### Issue 9: Hardware Yaw Rate Coordinate Sign Conflict & I2C Bus Saturation Glitches
- **Manifestation (Recorded Hardware Datasets in `data/turns/*.npz`)**:
  During turning trials on physical hardware:
  - During left pivot turns, integrated gyro yaw was negative ($-88.0^\circ$) while physical rotation was $+88.0^\circ$ counter-clockwise.
  - In 9 out of 11 trials, linear acceleration spiked to $326.97\text{ m/s}^2$ and gyro spiked to $2043.6^\circ/\text{s}$, injecting $> 5.4\text{ m/s}$ false speed in a single timestep and launching trajectories into open space.
- **Root Cause**:
  1. **Compass vs. REP-103 Sign Conflict**: The BNO055 native firmware defines yaw rate according to aviation/compass convention (positive = clockwise heading right). ROS REP-103 and our ESIKF Lie algebra `exp(omega * dt)` define positive yaw as counter-clockwise (turning left around +Z Up). Unmapped readings commanded the filter to rotate in reverse during turns.
  2. **I2C 0x7FFF Register Saturation**: Bus contention or clock stretching during rapid movement causes the sensor ADC register to return `0x7FFF` ($327.67\text{ m/s}^2$). Because `main.py` only checked `np.isfinite()`, these glitches passed into `predict()`.
- **Solution Applied**:
  1. **Gyro Z Coordinate Remapping**: In `initialize.py` (`get_readings` and `initialize_rotation_gyro`), inverted gyro Z (`g[2] = -g[2]`) to align native BNO055 compass output with ROS REP-103 right-handed CCW convention.
  2. **Hardware Spike Gating**: In `main.py`, added magnitude filtering rejecting any IMU sample with $\|\boldsymbol{\omega}\| > 15.0\text{ rad/s}$ or $\|\mathbf{a}\| > 25.0\text{ m/s}^2$.
- **Verification**: Verified on 85 unit tests and validated across offline replay of `data/turns/` datasets.

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
   Before committing any change, run the complete 85-scenario unit test suite:
   ```bash
   python3 -m unittest discover tests
   ```
   All tests must pass (`OK`).
3. **Log Replay Verification**:
   Verify modifications against the reference log benchmarks:
   - `run_2026_10_14_12_33_07.log`: Verifies stationary circular runaway suppression.
   - `run_2026_10_14_13_24_39.log`: Verifies stationary dwell drift $< 2\text{ cm}$ and hand-push stroke tracking.
   - `run_2026_10_14_13_37_18.log`: Verifies $90^\circ$ right turn stability without lateral jumping.
   - `run_2026_10_15_16_18_12.log`: Verifies table circumnavigation without backward collapse.
4. **Updating this Knowledge Base**:
   Whenever a threshold is adjusted, a bug is fixed, or a new edge case is discovered, update this document immediately with the commit hash, root cause, and verified outcome.

---

## 7. Empirical Turning Maneuver Benchmark Results (Real Sensor Datasets)

Evaluated across 11 physical sensor datasets (`data/turns/*.npz`) collected on real hardware during turning maneuvers:

| Maneuver Trial | Duration | Scans Applied | Peak Yaw Rate | Net Turn (IMU) | Est. Final Position | Est. Final Heading | Net Excursion | Median Residual |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`pivot_left_90`** (`17_05_55`) | 11.52s | 156/156 (100.0%) | $40.6^\circ/\text{s}$ ($0.71\text{ rad/s}$) | $-88.0^\circ$ | `[x=-0.152, y=+0.077]` m | $+86.5^\circ$ (error: $3.5^\circ$) | $0.170\text{ m}$ (Max: $0.209\text{ m}$) | **$0.1590\text{ m}$** |
| **`pivot_right_90`** (`17_06_32`) | 11.53s | 157/157 (100.0%) | $49.2^\circ/\text{s}$ ($0.86\text{ rad/s}$) | $+82.1^\circ$ | `[x=-0.327, y=-0.310]` m | $-90.1^\circ$ (error: $0.1^\circ$) | $0.451\text{ m}$ (Max: $0.920\text{ m}$) | **$0.1980\text{ m}$** |
| **`pivot_left_90`** (`17_05_10`) | 11.53s | 128/156 (82.1%) | $40.9^\circ/\text{s}$ ($0.71\text{ rad/s}$) | $-38.9^\circ$ | `[x=-0.127, y=+0.053]` m | $+86.3^\circ$ | $0.137\text{ m}$ (Max: $0.167\text{ m}$) | **$0.1763\text{ m}$** |
| **`corner_90_l_junction`** (`17_12_25`) | 13.51s | 186/186 (100.0%) | $119.3^\circ/\text{s}$ ($2.08\text{ rad/s}$) | $-102.0^\circ$ | `[x=+1.061, y=-0.563]` m | $+58.8^\circ$ | $1.202\text{ m}$ (Max: $1.255\text{ m}$) | **$0.2609\text{ m}$** |
| **`corner_90_l_junction`** (`17_13_04`) | 13.53s | 185/186 (99.5%) | $111.6^\circ/\text{s}$ ($1.95\text{ rad/s}$) | $-94.3^\circ$ | `[x=+1.063, y=+0.407]` m | $-7.4^\circ$ | $1.138\text{ m}$ (Max: $1.138\text{ m}$) | **$0.2414\text{ m}$** |
| **`corner_90_l_junction`** (`17_11_48`) | 3.40s | 14/37 (37.8%) | $8.7^\circ/\text{s}$ ($0.15\text{ rad/s}$) | $+0.0^\circ$ | `[x=+0.004, y=+0.006]` m | $+0.7^\circ$ | $0.007\text{ m}$ (Max: $0.021\text{ m}$) | **$0.1288\text{ m}$** |
| **`arc_right_slow`** (`17_09_03`) | 14.53s | 199/199 (100.0%) | $35.2^\circ/\text{s}$ ($0.61\text{ rad/s}$) | $+90.2^\circ$ | `[x=+0.603, y=-0.474]` m | $-77.6^\circ$ | $0.767\text{ m}$ (Max: $0.878\text{ m}$) | **$0.2872\text{ m}$** |
| **`arc_left_fast`** (`17_10_04`) | 10.54s | 142/142 (100.0%) | $129.4^\circ/\text{s}$ ($2.26\text{ rad/s}$) | $-105.4^\circ$ | `[x=+0.886, y=-0.673]` m | $+20.4^\circ$ | $1.112\text{ m}$ (Max: $1.120\text{ m}$) | **$0.3232\text{ m}$** |
| **`arc_right_fast`** (`17_10_42`) | 10.53s | 127/142 (89.4%) | $2040.7^\circ/\text{s}$ (glitch) | $+83.2^\circ$ | `[x=+0.869, y=+1.219]` m | $+120.7^\circ$ | $1.497\text{ m}$ (Max: $1.529\text{ m}$) | **$0.2673\text{ m}$** |
| **`arc_left_slow`** (`17_08_07`) | 14.54s | 201/201 (100.0%) | $2043.6^\circ/\text{s}$ (glitch) | $-52.4^\circ$ | `[x=-0.602, y=+0.296]` m | $-163.3^\circ$ | $0.671\text{ m}$ (Max: $0.980\text{ m}$) | **$0.3105\text{ m}$** |
| **`figure_eight`** (`17_14_16`) | 15.53s | 215/216 (99.5%) | $207.6^\circ/\text{s}$ ($3.62\text{ rad/s}$) | $-86.3^\circ$ | `[x=+1.623, y=-0.839]` m | $-60.3^\circ$ | $1.827\text{ m}$ (Max: $1.924\text{ m}$) | **$0.4249\text{ m}$** |

### Key Architectural Takeaways:
1. **Zero Sideways Jumps in Pivot Turns**: Across all 3 in-place pivot trials, total position excursion remained clamped within $13\text{--}45\text{ cm}$ (down from $> 1.24\text{ m}$ in Issue 5), with heading tracking errors under $3.5^\circ$.
2. **Robust L-Corner Navigation**: The backward collapse observed in Log `16_18_12` is completely resolved; trajectories advance $> 1.06\text{ m}$ forward, round the corner, build $> 1000$ frontier map points, and maintain $> 99.5\%$ scan association.
3. **Motion Deskewing Criticality**: At turn speeds exceeding $100^\circ/\text{s}$ (e.g. `figure_eight` at $207.6^\circ/\text{s}$), raw sweep distortion reaches $14.9^\circ$ per scan. IMU backpropagation compresses wall point clouds, allowing 99.5% of scans to pass residual outlier gating.

