"""
Aegis REST API — score network-flow style records.

  uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from features import FEATURE_COLUMNS, LABEL_MAP
    from infer import AegisEngine
except ImportError:
    from src.features import FEATURE_COLUMNS, LABEL_MAP
    from src.infer import AegisEngine

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
STATIC = ROOT / "static"
METRICS_PATH = ROOT / "reports" / "metrics.json"

app = FastAPI(
    title="Aegis Threat Detection",
    description="AI-assisted network flow anomaly & attack classification. Authorized use only.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: Optional[AegisEngine] = None


def get_engine() -> AegisEngine:
    global _engine
    if _engine is None:
        if not (MODELS / "random_forest.joblib").exists():
            raise HTTPException(503, "Models not trained. Run: python src/train.py")
        _engine = AegisEngine(str(MODELS))
    return _engine


class FlowRecord(BaseModel):
    duration: float = 0
    protocol_type: int = Field(0, description="0=tcp, 1=udp, 2=icmp")
    src_bytes: float = 0
    dst_bytes: float = 0
    wrong_fragment: float = 0
    urgent: float = 0
    hot: float = 0
    num_failed_logins: float = 0
    logged_in: float = 0
    num_compromised: float = 0
    root_shell: float = 0
    su_attempted: float = 0
    num_root: float = 0
    num_file_creations: float = 0
    num_shells: float = 0
    num_access_files: float = 0
    count: float = 0
    srv_count: float = 0
    serror_rate: float = 0
    srv_serror_rate: float = 0
    rerror_rate: float = 0
    srv_rerror_rate: float = 0
    same_srv_rate: float = 0
    diff_srv_rate: float = 0
    dst_host_count: float = 0
    dst_host_srv_count: float = 0
    dst_host_same_srv_rate: float = 0
    dst_host_diff_srv_rate: float = 0
    dst_host_serror_rate: float = 0
    dst_host_rerror_rate: float = 0


class ScoreRequest(BaseModel):
    flows: list[FlowRecord]


class ScoreOneRequest(BaseModel):
    flow: FlowRecord


@app.get("/api/health")
def health() -> dict[str, Any]:
    ready = (MODELS / "random_forest.joblib").exists()
    return {"status": "ok" if ready else "models_missing", "version": "1.0.0", "models_ready": ready}


@app.get("/api/metrics")
def metrics() -> Any:
    path = METRICS_PATH if METRICS_PATH.exists() else MODELS / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "No metrics yet — train first")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/schema")
def schema() -> dict[str, Any]:
    return {
        "features": FEATURE_COLUMNS,
        "labels": LABEL_MAP,
        "protocol_type": {"0": "tcp", "1": "udp", "2": "icmp"},
    }


@app.post("/api/score")
def score(req: ScoreRequest) -> dict[str, Any]:
    engine = get_engine()
    records = [f.model_dump() for f in req.flows]
    results = engine.score(records)
    return {"count": len(results), "results": results}


@app.post("/api/score/one")
def score_one(req: ScoreOneRequest) -> dict[str, Any]:
    engine = get_engine()
    return engine.score(req.flow.model_dump())[0]


@app.get("/api/demo-samples")
def demo_samples() -> dict[str, Any]:
    """Curated examples for the dashboard."""
    return {
        "samples": [
            {
                "name": "Normal web session",
                "expected": "benign",
                "flow": {
                    "duration": 2.1,
                    "protocol_type": 0,
                    "src_bytes": 1200,
                    "dst_bytes": 4500,
                    "logged_in": 1,
                    "count": 6,
                    "srv_count": 5,
                    "same_srv_rate": 0.95,
                    "dst_host_count": 12,
                    "dst_host_same_srv_rate": 0.9,
                },
            },
            {
                "name": "Port sweep / probe",
                "expected": "probe",
                "flow": {
                    "duration": 0.2,
                    "protocol_type": 0,
                    "src_bytes": 20,
                    "dst_bytes": 0,
                    "count": 120,
                    "srv_count": 100,
                    "same_srv_rate": 0.1,
                    "diff_srv_rate": 0.85,
                    "dst_host_count": 200,
                    "dst_host_diff_srv_rate": 0.7,
                },
            },
            {
                "name": "High-volume DoS-like",
                "expected": "dos",
                "flow": {
                    "duration": 0.05,
                    "protocol_type": 0,
                    "src_bytes": 80000,
                    "dst_bytes": 40,
                    "count": 300,
                    "srv_count": 280,
                    "serror_rate": 0.9,
                    "srv_serror_rate": 0.85,
                    "dst_host_serror_rate": 0.8,
                },
            },
            {
                "name": "Brute-force login pattern",
                "expected": "r2l",
                "flow": {
                    "duration": 1.0,
                    "protocol_type": 0,
                    "src_bytes": 80,
                    "dst_bytes": 120,
                    "num_failed_logins": 5,
                    "logged_in": 0,
                    "count": 15,
                    "srv_count": 12,
                },
            },
            {
                "name": "Privilege escalation indicators",
                "expected": "u2r",
                "flow": {
                    "duration": 3.0,
                    "protocol_type": 0,
                    "src_bytes": 400,
                    "dst_bytes": 800,
                    "hot": 4,
                    "num_compromised": 2,
                    "root_shell": 1,
                    "su_attempted": 1,
                    "num_root": 2,
                    "num_file_creations": 3,
                    "logged_in": 1,
                },
            },
        ]
    }


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = STATIC / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Dashboard not found")
    return FileResponse(index_path)
