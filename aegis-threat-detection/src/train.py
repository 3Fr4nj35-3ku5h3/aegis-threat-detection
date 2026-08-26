"""
Train Aegis models:
  1) IsolationForest — unsupervised anomaly score
  2) RandomForest    — supervised attack-category classifier
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from features import FEATURE_COLUMNS, LABEL_MAP
except ImportError:
    from src.features import FEATURE_COLUMNS, LABEL_MAP


def train(data_path: str, model_dir: str, seed: int = 42) -> dict:
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y_multi = df["label"].values
    y_bin = df["is_attack"].values

    X_train, X_test, y_multi_tr, y_multi_te, y_bin_tr, y_bin_te = train_test_split(
        X, y_multi, y_bin, test_size=0.25, random_state=seed, stratify=y_multi
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Unsupervised anomaly detector (fit on benign only) ---
    benign_mask = y_bin_tr == 0
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(X_train_s[benign_mask])

    # anomaly score: higher = more anomalous (invert sklearn score)
    iso_scores = -iso.score_samples(X_test_s)
    iso_pred = (iso.predict(X_test_s) == -1).astype(int)
    iso_auc = roc_auc_score(y_bin_te, iso_scores)
    iso_ap = average_precision_score(y_bin_te, iso_scores)

    # --- Supervised multi-class ---
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_multi_tr)
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)

    report = classification_report(
        y_multi_te,
        y_pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_multi_te, y_pred).tolist()

    # Binary attack detection from supervised (any non-benign)
    bin_pred = (y_pred > 0).astype(int)
    # probability of attack = 1 - P(benign)
    classes = list(clf.classes_)
    benign_idx = classes.index(0) if 0 in classes else 0
    attack_proba = 1.0 - y_proba[:, benign_idx]
    sup_auc = roc_auc_score(y_bin_te, attack_proba)
    sup_ap = average_precision_score(y_bin_te, attack_proba)

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(iso, out / "isolation_forest.joblib")
    joblib.dump(clf, out / "random_forest.joblib")

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "isolation_forest": {
            "roc_auc_attack": round(float(iso_auc), 4),
            "avg_precision_attack": round(float(iso_ap), 4),
            "anomaly_rate_on_test": round(float(iso_pred.mean()), 4),
        },
        "random_forest": {
            "roc_auc_attack": round(float(sup_auc), 4),
            "avg_precision_attack": round(float(sup_ap), 4),
            "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
            "per_class": {
                k: {
                    "precision": round(v["precision"], 4),
                    "recall": round(v["recall"], 4),
                    "f1": round(v["f1-score"], 4),
                    "support": int(v["support"]),
                }
                for k, v in report.items()
                if k in LABEL_MAP.values()
            },
            "confusion_matrix": cm,
        },
        "feature_importance_top10": sorted(
            [
                {"feature": FEATURE_COLUMNS[i], "importance": round(float(imp), 4)}
                for i, imp in enumerate(clf.feature_importances_)
            ],
            key=lambda d: d["importance"],
            reverse=True,
        )[:10],
    }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/flows.csv")
    parser.add_argument("--models", default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.data).exists():
        print("[!] No data found — generating synthetic set…")
        from generate_data import generate

        Path(args.data).parent.mkdir(parents=True, exist_ok=True)
        generate(12000, args.seed).to_csv(args.data, index=False)

    metrics = train(args.data, args.models, args.seed)
    print(json.dumps(metrics, indent=2))
    print(f"\n[+] Models saved under {args.models}/")


if __name__ == "__main__":
    main()
