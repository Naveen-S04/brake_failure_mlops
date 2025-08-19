import argparse, os, json, joblib, numpy as np, mlflow
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from pathlib import Path
from src.common.io import load_params

def main(params_path: str):
    params = load_params(params_path)
    p = params["paths"]
    mlp = params.get("mlflow", {})
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or mlp.get("tracking_uri") or "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    X_test = np.load(p["features_test"])
    y_test = np.load(p["labels_test"])
    model_path = Path(p["model_dir"]) / p["model_name"]
    model = joblib.load(model_path)

    proba = model.predict_proba(X_test)[:,1]
    preds = (proba >= params["evaluation"]["threshold"]).astype(int)

    auc = roc_auc_score(y_test, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)

    report = {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    out = Path(params["evaluation"]["output_report"])
    out.write_text(json.dumps(report, indent=2))
    with mlflow.start_run():
        mlflow.log_metrics(report)
        mlflow.log_artifact(out.as_posix(), artifact_path="eval")

    print("Evaluation metrics:", report)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args = ap.parse_args()
    main(args.params)