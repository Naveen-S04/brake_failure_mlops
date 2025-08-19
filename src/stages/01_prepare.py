import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse
import yaml

from src.common.io import load_params

def main(params_path: str):
    params = load_params(params_path)
    test_size = params["prepare"]["test_size"]
    random_state = params["prepare"]["random_state"]


    os.makedirs("data/processed", exist_ok=True)

    # load raw data
    df = pd.read_csv("data/raw/brake_sensor_data.csv")

    # split into train/test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # save outputs
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Wrote {train_path} -> shape={train_df.shape}")
    print(f"✅ Wrote {test_path} -> shape={test_df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)
