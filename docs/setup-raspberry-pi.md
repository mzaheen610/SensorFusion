# Raspberry Pi Setup

## Hardware Assumptions

- Raspberry Pi with Python 3 available.
- RPLidar connected as `/dev/ttyUSB0`.
- ICM20948 IMU connected over I2C at address `0x69`.
- Camera smoke test uses `picamzero` and a Raspberry Pi camera stack.

## Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The dependency file currently includes:

- `matplotlib`
- `numpy`
- `rplidar-roboticia`
- `adafruit-blinka`
- `adafruit-circuitpython-icm20x`
- `smbus2`

Camera testing may also require `picamzero`, depending on the Pi image.

## Running SLAM

```bash
mkdir -p outputs
python scripts/run_headless_slam.py --output-prefix outputs/fusion_output
```

The canonical script is intended for headless operation over SSH. It uses Matplotlib's `Agg` backend and periodically writes map images instead of opening a desktop window.

## Hardware Smoke Tests

Run these only on the Raspberry Pi with the relevant device connected:

```bash
python tests/hardware/test_lidar.py
python tests/hardware/test_imu.py
python tests/hardware/test_camera.py
```

The LiDAR and IMU tests are blocking hardware checks. Stop them with `Ctrl+C` when enough samples have been observed.

## Notes

- If the LiDAR port differs, update the script argument or constant before running.
- If I2C is disabled, enable it using `raspi-config` and reboot.
- Generated map output should go under `outputs/`; that directory is intentionally ignored by Git.
