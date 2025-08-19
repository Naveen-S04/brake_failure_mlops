# save this as src/stages/00_generate_data.py
import pandas as pd
import numpy as np
import os

os.makedirs("data/raw", exist_ok=True)

n_samples = 5000
data = {
    "speed": np.random.randint(0, 200, n_samples),
    "brake_pressure": np.random.uniform(0, 1, n_samples),
    "temperature": np.random.randint(20, 120, n_samples),
    "vibration": np.random.uniform(0, 5, n_samples),
    "pad_wear": np.random.uniform(0, 10, n_samples),
    "disk_wear": np.random.uniform(0, 10, n_samples),
    "humidity": np.random.randint(20, 100, n_samples),
    "load": np.random.randint(100, 1000, n_samples),
    "fluid_level": np.random.uniform(0, 1, n_samples),
    "failure": np.random.randint(0, 2, n_samples),
}

df = pd.DataFrame(data)
df.to_csv("data/raw/brake_sensor_data.csv", index=False)

print("✅ brake_sensor_data.csv generated at data/raw/")
