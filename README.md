# SensorFusion

SensorFusion is an embedded Raspberry Pi prototype for LiDAR, IMU, and camera based navigation research. The current working path focuses on real-time 2D LiDAR plus IMU fusion for headless SLAM map generation.

The repo is organized as an engineering handoff for the Embedded Application Lab project: runnable scripts live in `scripts/`, hardware smoke tests live in `tests/hardware/`, older prototypes are preserved in `experiments/archived/`, and project knowledge is mirrored from Notion into `docs/`.

## Current Status

- Canonical runnable implementation: `scripts/run_headless_slam.py`
- Target platform: Raspberry Pi with RPLidar and ICM20948 IMU
- Runtime mode: headless Matplotlib `Agg` backend with periodic PNG map output
- Main estimation path: IMU prediction, ICP scan matching correction, local submap, and occupancy grid mapping
- Validation state: hardware smoke tests exist, but no automated quantitative accuracy benchmark is included yet

## Layout

```text
.
├── scripts/                 # Runnable entrypoints for Raspberry Pi execution
├── src/sensorfusion/         # Future reusable Python package modules
├── tests/hardware/          # Hardware smoke tests; run only on target hardware
├── experiments/archived/    # Historical prototypes and older model variants
├── docs/                    # Setup, architecture, research, and troubleshooting notes
├── assets/examples/         # Small checked-in example images
└── outputs/                 # Generated maps and runtime outputs, ignored by Git
```

## Quick Start

Create and activate an environment on the Raspberry Pi:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the canonical headless SLAM script:

```bash
mkdir -p outputs
python scripts/run_headless_slam.py --output-prefix outputs/fusion_output
```

The script expects the LiDAR on `/dev/ttyUSB0`, an ICM20948 IMU at I2C address `0x69`, and writes map images during runtime. See `docs/setup-raspberry-pi.md` and `docs/troubleshooting.md` before running on new hardware.

## Documentation

- `docs/architecture.md` explains the data flow and estimator/mapping stages.
- `docs/setup-raspberry-pi.md` covers environment and hardware setup.
- `docs/research-summary.md` mirrors the key Notion research and progress findings.
- `docs/experiments.md` explains the archived prototypes and canonical script choice.
- `docs/troubleshooting.md` captures common Raspberry Pi, LiDAR, IMU, and headless plotting issues.
