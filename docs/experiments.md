# Experiments And Archived Prototypes

The repository previously kept all scripts at the top level. During cleanup, historical variants were moved to `experiments/archived/` so they remain available without competing with the canonical implementation.

## Canonical Script

Use:

```bash
python scripts/run_headless_slam.py
```

This script was selected because it contains the most complete fusion and mapping behavior currently present in the project:

- 8-state planar EKF.
- IMU bias handling.
- Tilt compensation.
- Stationary detection and zero-velocity update.
- ICP gating.
- Local submap management.
- Occupancy grid free-space ray updates.

## Archived Files

- `experiments/archived/fusion_store_map.py`: earlier headless save-to-disk implementation with a simpler EKF and endpoint-only map updates.
- `experiments/archived/lidarIMU.py`: earlier interactive or robust LiDAR plus IMU fusion variant.
- `experiments/archived/test.py`, `test2.py`, `test3.py`: prototype EKF/fusion iterations.
- `experiments/archived/updated/full_state_model.py`: older full-state variant kept for history; it appears inconsistent with the current canonical script.

## Policy

- Do not build new features on archived variants.
- If an archived file contains a useful idea, port the idea into the canonical path intentionally and document the change.
- Hardware smoke tests belong in `tests/hardware/`, not in `experiments/archived/`.
