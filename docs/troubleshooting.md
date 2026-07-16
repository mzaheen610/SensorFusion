# Troubleshooting

## LiDAR Connection

The current scripts assume the LiDAR is available at `/dev/ttyUSB0` with baud rate `256000`.

Common checks:

```bash
ls /dev/ttyUSB*
python tests/hardware/test_lidar.py
```

The RPLidar library used in this project has shown a health-check unpacking issue. The project scripts monkey-patch `get_health()` to return a good status before scanning. Keep this documented if changing LiDAR libraries.

## IMU Connection

The IMU path assumes an ICM20948 on I2C address `0x69`.

Common checks:

```bash
i2cdetect -y 1
python tests/hardware/test_imu.py
```

If no address appears, confirm I2C is enabled on the Pi and wiring matches `SCL`, `SDA`, `3V3`, and `GND`.

## Headless Plotting

The canonical runtime uses Matplotlib `Agg`, which is appropriate for SSH and no-monitor operation. Avoid switching the canonical script to `TkAgg` unless the target environment has a desktop session.

If map files are not changing:

- Confirm LiDAR scans are being received.
- Check console output for scan initialization messages.
- Confirm the process has write access to the output path.

## Generated Outputs

Runtime maps and logs should be kept in `outputs/`. This directory is ignored by Git so repeated runs do not pollute commits.

Use an output prefix inside that directory:

```bash
python scripts/run_headless_slam.py --output-prefix outputs/fusion_output
```

## Hardware Tests

The scripts in `tests/hardware/` are not automated unit tests. They touch real devices and may block while reading samples. Run them manually on the Raspberry Pi only.
