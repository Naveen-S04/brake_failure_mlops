import argparse
import pandas as pd
import numpy as np
import os
from src.common.io import load_params
from sklearn.model_selection import train_test_split

def main(config_path):
    params = load_params(config_path)
    
    # Ensure output folder exists
    os.makedirs("data/features", exist_ok=True)
    
    # Load processed train/test data
    train_df = pd.read_csv(params["paths"]["processed_data"])
    test_df = pd.read_csv(params["paths"]["processed_data"].replace("train", "test"))
    
    # Features & target
    features = params["features"]["numeric"]
    target = "failure"
    
    X_train = train_df[features].copy()
    y_train = train_df[[target]].copy()
    
    X_test = test_df[features].copy()
    y_test = test_df[[target]].copy()
    
    # Save as CSV for DVC tracking
    X_train.to_csv("data/features/train_features.csv", index=False)
    X_test.to_csv("data/features/test_features.csv", index=False)
    y_train.to_csv("data/features/train_labels.csv", index=False)
    y_test.to_csv("data/features/test_labels.csv", index=False)
    
    # Save as .npy
    np.save(params["paths"]["features_train"], X_train.values)
    np.save(params["paths"]["features_test"], X_test.values)
    np.save(params["paths"]["labels_train"], y_train.values)
    np.save(params["paths"]["labels_test"], y_test.values)
    
    print("Feature engineering complete. CSV and .npy files saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    args = parser.parse_args()
    
    main(args.params)
