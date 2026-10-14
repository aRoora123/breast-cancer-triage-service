
import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import yaml

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/raw/breast_cancer.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model-version", default="1.0.0")
    args = parser.parse_args()

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    seed = params["seed"]
    df = pd.read_csv(args.data_path)

    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names]
    # Dataset target: 0 = malignant, 1 = benign.
    # Convert malignant into the positive class (1).
    y = (df["target"] == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["split"]["test_size"],
        random_state=seed,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=params["cross_validation"]["folds"],
        shuffle=True,
        random_state=seed,
    )

    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=params["model"]["logistic_regression"]["max_iter"],
                class_weight=params["model"]["logistic_regression"]["class_weight"],
                random_state=seed,
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=params["model"]["random_forest"]["n_estimators"],
                max_depth=params["model"]["random_forest"]["max_depth"],
                class_weight=params["model"]["random_forest"]["class_weight"],
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }

    scoring = {
        "malignant_recall": "recall",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }

    results = {}
    best_name = None
    best_model = None
    best_recall = -1

    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        mean_scores = {
            metric: float(scores[f"test_{metric}"].mean())
            for metric in scoring
        }
        results[name] = {"cv_mean": mean_scores}
        print(f"\n{name} CV means:")
        for metric, value in mean_scores.items():
            print(f"  {metric}: {value:.4f}")

        if mean_scores["malignant_recall"] > best_recall:
            best_recall = mean_scores["malignant_recall"]
            best_name = name
            best_model = model

    best_model.fit(X_train, y_train)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    test_metrics = {
        "malignant_recall": float(recall_score(y_test, predictions)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    results[best_name]["test"] = test_metrics
    print(f"\nSelected model: {best_name}")
    print("Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_dir / "model.joblib")

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_sha = "unknown"

    model_card = {
        "model_version": args.model_version,
        "model_family": best_name,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "seed": seed,
        "scikit_learn_version": sklearn.__version__,
        "decision_threshold": 0.5,
        "cross_validation": results[best_name]["cv_mean"],
        "test_metrics": test_metrics,
        "git_commit": git_sha,
        "candidate_results": results,
    }

    with open(output_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model and model card to: {output_dir}")

    if test_metrics["malignant_recall"] < 0.95:
        print("ERROR: Test malignant recall is below 0.95.")
        sys.exit(1)


if __name__ == "__main__":
    main()
