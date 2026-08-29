"""
Aegis REST API v1.2 — triage, batch CSV, audit log, model card.

  uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import csv
import io
import json
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
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
SAMPLE_CSV = ROOT / "static" / "sample_flows.csv"

app = FastAPI(
    title="Aegis Console",
    description="AI-assisted network flow triage for security analysts. Authorized use only.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: Optional[AegisEngine] = None
_audit_lock = threading.Lock()
_audit: deque[dict[str, Any]] = deque(maxlen=200)


def get_engine() -> AegisEngine:
    global _engine
    if _engine is None:
        if not (MODELS / "random_forest.joblib").exists():
            raise HTTPException(503, "Models not trained. Run: python src/train.py")
        _engine = AegisEngine(str(MODELS))
    return _engine


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _audit_add(entry: dict[str, Any]) -> None:
    with _audit_lock:
        _audit.appendleft(entry)


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
    sensitivity: float = Field(0.0, ge=-1.0, le=1.0)
    source: str = "api"


class ScoreOneRequest(BaseModel):
    flow: FlowRecord
    sensitivity: float = Field(0.0, ge=-1.0, le=1.0)
    source: str = "console"


@app.get("/api/health")
def health() -> dict[str, Any]:
    ready = (MODELS / "random_forest.joblib").exists()
    return {
        "status": "ok" if ready else "models_missing",
        "version": "1.2.0",
        "models_ready": ready,
    }


@app.get("/api/metrics")
def metrics() -> Any:
    path = METRICS_PATH if METRICS_PATH.exists() else MODELS / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "No metrics yet — train first")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/model-card")
def model_card() -> dict[str, Any]:
    metrics_data = None
    path = METRICS_PATH if METRICS_PATH.exists() else MODELS / "metrics.json"
    if path.exists():
        metrics_data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": "Aegis Console",
        "version": "1.2.0",
        "purpose": "Decision support for network-flow triage by security analysts.",
        "pipeline": [
            "StandardScaler on 30 flow features",
            "Isolation Forest (anomaly vs benign baseline)",
            "Random Forest (benign / probe / dos / r2l / u2r)",
            "Fused risk score with adjustable sensitivity",
            "Analyst playbook enrichment (no auto-block)",
        ],
        "training_data": {
            "type": "synthetic KDD-style flows",
            "reason": "Reproducible demo without redistributing third-party datasets",
            "limitation": "Not calibrated on your network — retrain before production use",
        },
        "labels": LABEL_MAP,
        "features": FEATURE_COLUMNS,
        "sensitivity": {
            "range": [-1, 1],
            "default": 0,
            "meaning": "-1 fewer alerts (higher thresholds); +1 more alerts (lower thresholds)",
        },
        "not_for": [
            "Autonomous blocking or inline prevention",
            "Unauthorized scanning of systems you do not own",
            "Sole evidence for legal or HR action without corroboration",
        ],
        "metrics": metrics_data,
    }


@app.get("/api/schema")
def schema() -> dict[str, Any]:
    return {
        "features": FEATURE_COLUMNS,
        "labels": LABEL_MAP,
        "protocol_type": {"0": "tcp", "1": "udp", "2": "icmp"},
        "csv_columns": FEATURE_COLUMNS,
    }


@app.get("/api/audit")
def audit_log(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(limit, 200))
    with _audit_lock:
        items = list(_audit)[:limit]
    return {"count": len(items), "entries": items}


@app.delete("/api/audit")
def audit_clear() -> dict[str, str]:
    with _audit_lock:
        _audit.clear()
    return {"status": "cleared"}


@app.post("/api/score")
def score(req: ScoreRequest) -> dict[str, Any]:
    engine = get_engine()
    records = [f.model_dump() for f in req.flows]
    if len(records) > 500:
        raise HTTPException(400, "Max 500 flows per request")
    results = engine.score(records, sensitivity=req.sensitivity)
    batch_id = str(uuid.uuid4())[:8]
    for i, r in enumerate(results):
        _audit_add(
            {
                "id": f"{batch_id}-{i}",
                "ts": _utc_now(),
                "source": req.source,
                "predicted_class": r.get("predicted_class"),
                "risk_level": r.get("risk_level"),
                "risk_score": r.get("risk_score"),
                "disposition": (r.get("analyst") or {}).get("disposition"),
                "sensitivity": req.sensitivity,
            }
        )
    return {"count": len(results), "batch_id": batch_id, "results": results}


@app.post("/api/score/one")
def score_one(req: ScoreOneRequest) -> dict[str, Any]:
    engine = get_engine()
    result = engine.score(req.flow.model_dump(), sensitivity=req.sensitivity)[0]
    entry_id = str(uuid.uuid4())[:8]
    _audit_add(
        {
            "id": entry_id,
            "ts": _utc_now(),
            "source": req.source,
            "predicted_class": result.get("predicted_class"),
            "risk_level": result.get("risk_level"),
            "risk_score": result.get("risk_score"),
            "disposition": (result.get("analyst") or {}).get("disposition"),
            "sensitivity": req.sensitivity,
        }
    )
    result["audit_id"] = entry_id
    return result


@app.post("/api/score/csv")
async def score_csv(
    file: UploadFile = File(...),
    sensitivity: float = 0.0,
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file")
    raw = await file.read()
    if len(raw) > 2_000_000:
        raise HTTPException(400, "CSV too large (max ~2MB)")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(400, "CSV has no header row")
    records: list[dict[str, Any]] = []
    for row in reader:
        rec: dict[str, Any] = {}
        for col in FEATURE_COLUMNS:
            val = row.get(col, 0)
            try:
                rec[col] = float(val) if val not in (None, "") else 0.0
            except ValueError:
                rec[col] = 0.0
        records.append(rec)
        if len(records) >= 500:
            break
    if not records:
        raise HTTPException(400, "No data rows in CSV")
    engine = get_engine()
    sens = float(max(-1.0, min(1.0, sensitivity)))
    results = engine.score(records, sensitivity=sens)
    batch_id = str(uuid.uuid4())[:8]
    for i, r in enumerate(results):
        _audit_add(
            {
                "id": f"{batch_id}-{i}",
                "ts": _utc_now(),
                "source": "csv",
                "predicted_class": r.get("predicted_class"),
                "risk_level": r.get("risk_level"),
                "risk_score": r.get("risk_score"),
                "disposition": (r.get("analyst") or {}).get("disposition"),
                "sensitivity": sens,
            }
        )
    # summary counts
    levels: dict[str, int] = {}
    classes: dict[str, int] = {}
    for r in results:
        levels[r["risk_level"]] = levels.get(r["risk_level"], 0) + 1
        classes[r["predicted_class"]] = classes.get(r["predicted_class"], 0) + 1
    return {
        "batch_id": batch_id,
        "count": len(results),
        "sensitivity": sens,
        "summary": {"risk_levels": levels, "classes": classes},
        "results": results,
    }


@app.get("/api/sample-csv")
def sample_csv() -> PlainTextResponse:
    """Downloadable sample for batch triage demos."""
    header = ",".join(FEATURE_COLUMNS)
    # compact demo rows aligned with console scenarios
    rows = [
        # benign-ish
        "2.1,0,1200,4500,0,0,0,0,1,0,0,0,0,0,0,0,6,5,0,0,0,0,0.95,0.05,12,10,0.9,0.1,0,0",
        # probe-ish
        "0.2,0,20,0,0,0,0,0,0,0,0,0,0,0,0,0,120,100,0,0,0,0,0.1,0.85,200,50,0.2,0.7,0,0",
        # dos-ish
        "0.05,0,80000,40,0,0,0,0,0,0,0,0,0,0,0,0,300,280,0.9,0.85,0,0,0.5,0.2,100,80,0.4,0.2,0.8,0",
    ]
    body = header + "\n" + "\n".join(rows) + "\n"
    return PlainTextResponse(
        body,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=aegis_sample_flows.csv"},
    )


@app.get("/api/demo-samples")
def demo_samples() -> dict[str, Any]:
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
