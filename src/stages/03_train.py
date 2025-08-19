import argparse, os, joblib, numpy as np, mlflow, mlflow.sklearn
from xgboost import XGBClassifier
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.common.io import load_params

def main(params_path: str):
    params = load_params(params_path)
    p = params["paths"]
    m = params["model"]
    mlp = params.get("mlflow", {})
    exp_name = mlp.get("experiment_name", "brake-failure-exp")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or mlp.get("tracking_uri") or "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    X_train = np.load(p["features_train"])
    y_train = np.load(p["labels_train"])

    if m["type"] == "xgboost":
        model = XGBClassifier(**m["params"], eval_metric="logloss")
    else:
        raise ValueError("Unsupported model type")

    with mlflow.start_run():
        model.fit(X_train, y_train)
        # simple metric on train just to log something
        proba = model.predict_proba(X_train)[:,1]
        auc = roc_auc_score(y_train, proba)
        mlflow.log_metric("train_auc", float(auc))

        Path(p["model_dir"]).mkdir(parents=True, exist_ok=True)
        model_path = Path(p["model_dir"]) / p["model_name"]
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path.as_posix(), artifact_path="model")
        mlflow.log_param("model_type", m["type"])
        mlflow.log_params(m["params"])

        print(f"Saved model -> {model_path}")
    print("Training done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args = ap.parse_args()
    main(args.params)