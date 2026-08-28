"""Load trained models and score a single flow or batch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from features import FEATURE_COLUMNS, LABEL_MAP
    from playbook import enrich
except ImportError:
    from src.features import FEATURE_COLUMNS, LABEL_MAP
    from src.playbook import enrich


class AegisEngine:
    def __init__(self, model_dir: str = "models"):
        root = Path(model_dir)
        self.scaler = joblib.load(root / "scaler.joblib")
        self.iso = joblib.load(root / "isolation_forest.joblib")
        self.clf = joblib.load(root / "random_forest.joblib")
        self.classes = list(self.clf.classes_)

    def _matrix(self, records: list[dict[str, Any]] | dict[str, Any]) -> np.ndarray:
        if isinstance(records, dict):
            records = [records]
        df = pd.DataFrame(records)
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        return df[FEATURE_COLUMNS].astype(np.float64).values

    def score(self, records: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
        X = self._matrix(records)
        Xs = self.scaler.transform(X)

        iso_raw = -self.iso.score_samples(Xs)
        iso_flag = (self.iso.predict(Xs) == -1).astype(int)

        proba = self.clf.predict_proba(Xs)
        pred = self.clf.predict(Xs)
        benign_idx = self.classes.index(0) if 0 in self.classes else 0
        attack_prob = 1.0 - proba[:, benign_idx]

        out = []
        for i in range(len(X)):
            class_id = int(pred[i])
            class_probs = {
                LABEL_MAP.get(int(c), str(c)): round(float(proba[i, j]), 4)
                for j, c in enumerate(self.classes)
            }
            # Combined risk: blend supervised attack prob + normalized anomaly
            anomaly_norm = float(1 / (1 + np.exp(-(iso_raw[i] - 0.5) * 3)))  # soft squash
            risk = round(float(0.65 * attack_prob[i] + 0.35 * anomaly_norm), 4)
            if risk >= 0.75:
                level = "critical"
            elif risk >= 0.5:
                level = "high"
            elif risk >= 0.3:
                level = "medium"
            else:
                level = "low"

            raw = {
                "predicted_class": LABEL_MAP.get(class_id, "unknown"),
                "predicted_class_id": class_id,
                "class_probabilities": class_probs,
                "attack_probability": round(float(attack_prob[i]), 4),
                "anomaly_score": round(float(iso_raw[i]), 4),
                "anomaly_flag": bool(iso_flag[i]),
                "risk_score": risk,
                "risk_level": level,
                "model_version": "1.1.0",
            }
            out.append(enrich(raw))
        return out
