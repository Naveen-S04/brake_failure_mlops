import numpy as np
import pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)

N = 5000
speed = rng.normal(60, 20, N).clip(0, 180)
brake_pedal_pressure = rng.uniform(0, 1, N)
rotor_temp = rng.normal(120, 40, N).clip(20, 600)
vibration = rng.normal(0.1, 0.05, N).clip(0, 1)
battery_voltage = rng.normal(12.6, 0.6, N).clip(10, 14.8)
pad_thickness = rng.normal(8, 2.5, N).clip(0.5, 15)
mileage = rng.integers(1000, 200000, N)
ambient_temp = rng.normal(25, 10, N).clip(-10, 55)
humidity = rng.uniform(0.1, 0.95, N)

# Hidden rule to create target probability
risk = (
    0.003*(speed-80).clip(min=0) +
    1.2*brake_pedal_pressure +
    0.004*(rotor_temp-140).clip(min=0) +
    1.8*vibration +
    0.05*(13.0 - battery_voltage).clip(min=0) +
    0.08*(7 - pad_thickness).clip(min=0) +
    0.000008*mileage +
    0.01*(ambient_temp-35).clip(min=0) +
    0.2*humidity
)

prob = 1 / (1 + np.exp(-(risk - 3.5)))
failure = (rng.uniform(0,1,N) < prob).astype(int)

df = pd.DataFrame({
    "speed": speed,
    "brake_pedal_pressure": brake_pedal_pressure,
    "rotor_temp": rotor_temp,
    "vibration": vibration,
    "battery_voltage": battery_voltage,
    "pad_thickness": pad_thickness,
    "mileage": mileage,
    "ambient_temp": ambient_temp,
    "humidity": humidity,
    "brake_failure": failure
})

Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/brake_sensor_data.csv", index=False)
print("Wrote data/raw/brake_sensor_data.csv", df.shape)