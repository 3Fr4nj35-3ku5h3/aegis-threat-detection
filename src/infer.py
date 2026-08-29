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

MODEL_VERSION = "1.2.0"


def _risk_level(risk: float, sensitivity: float) -> str:
    """
    sensitivity in [-1, 1]:
      -1 = fewer alerts (higher bars for high/critical)
       0 = default
      +1 = more alerts (lower bars)
    """
    s = float(np.clip(sensitivity, -1.0, 1.0))
    # shift thresholds: positive sensitivity lowers bars
    t_med = 0.30 - 0.12 * s
    t_high = 0.50 - 0.12 * s
    t_crit = 0.75 - 0.10 * s
    if risk >= t_crit:
        return "critical"
    if risk >= t_high:
        return "high"
    if risk >= t_med:
        return "medium"
    return "low"


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

    def score(
        self,
        records: list[dict[str, Any]] | dict[str, Any],
        sensitivity: float = 0.0,
    ) -> list[dict[str, Any]]:
        X = self._matrix(records)
        Xs = self.scaler.transform(X)

        iso_raw = -self.iso.score_samples(Xs)
        iso_flag = (self.iso.predict(Xs) == -1).astype(int)

        proba = self.clf.predict_proba(Xs)
        pred = self.clf.predict(Xs)
        benign_idx = self.classes.index(0) if 0 in self.classes else 0
        attack_prob = 1.0 - proba[:, benign_idx]

        out: list[dict[str, Any]] = []
        for i in range(len(X)):
            class_id = int(pred[i])
            class_probs = {
                LABEL_MAP.get(int(c), str(c)): round(float(proba[i, j]), 4)
                for j, c in enumerate(self.classes)
            }
            anomaly_norm = float(1 / (1 + np.exp(-(iso_raw[i] - 0.5) * 3)))
            risk = round(float(0.65 * attack_prob[i] + 0.35 * anomaly_norm), 4)
            level = _risk_level(risk, sensitivity)

            raw = {
                "predicted_class": LABEL_MAP.get(class_id, "unknown"),
                "predicted_class_id": class_id,
                "class_probabilities": class_probs,
                "attack_probability": round(float(attack_prob[i]), 4),
                "anomaly_score": round(float(iso_raw[i]), 4),
                "anomaly_flag": bool(iso_flag[i]),
                "risk_score": risk,
                "risk_level": level,
                "sensitivity": round(float(sensitivity), 3),
                "model_version": MODEL_VERSION,
            }
            out.append(enrich(raw))
        return out
