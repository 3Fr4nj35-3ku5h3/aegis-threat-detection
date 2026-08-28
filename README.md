# Aegis Console

**AI-assisted network flow triage for security analysts.**

Aegis is a **decision-support product**, not a toy demo. It scores network-flow style records, explains the result, and recommends human next steps. It does **not** auto-block traffic.

## Who it is for

| Role | How they use Aegis |
|------|---------------------|
| **SOC / network analyst** | Triage suspicious flows; get priority, class, and playbook actions |
| **Detection engineer** | Inspect model metrics, feature importance, and API contracts |
| **Hiring manager** | See end-to-end security + ML product thinking |

## What it does

1. **Unsupervised anomaly** — Isolation Forest vs a benign baseline
2. **Supervised classification** — benign · probe · dos · r2l · u2r
3. **Fused risk score** — low / medium / high / critical
4. **Analyst packet** — summary, rationale, recommended actions, disposition

### Design principles

- **Assist, don’t automate harm** — no auto-block in this product
- **Explainability first** — class probabilities + written rationale
- **Honest scope** — synthetic training data for reproducibility; production needs your telemetry
- **Operational language** — priority, disposition, playbooks — not “AI magic”

## Quick start

```bash
git clone https://github.com/3Fr4nj35-3ku5h3/aegis-threat-detection.git
cd aegis-threat-detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# If models/ is empty:
python src/generate_data.py --n 12000
python src/train.py
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000** → select a scenario → **Run triage**.

### API

| Endpoint | Purpose |
|----------|---------|
| `GET /api/health` | Liveness + model readiness |
| `GET /api/metrics` | Held-out metrics (model card) |
| `GET /api/schema` | Feature + label schema |
| `POST /api/score/one` | Score a single flow → analyst packet |
| `POST /api/score` | Batch score |
| `GET /api/demo-samples` | Built-in triage scenarios |

## Deploy (Render)

```
Build:  pip install -r requirements.txt
Start:  uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

Use the included `Procfile`. Ensure `models/*.joblib` are in the repo.

## Layout

```
src/          generate, train, infer, playbook, api, features
static/       Aegis Console UI
models/       joblib artifacts
reports/      metrics.json
Procfile
```

## Model card

Trained on synthetic KDD-style features for demo reproducibility. Publish metrics from `reports/metrics.json` after training.

Before production: use your labeled flows, recalibrate thresholds with analysts, log scores for audit, keep a human in the loop for high/critical.

## Legal

Authorized environments only. Unauthorized access to systems is illegal.

## Author

**Francis Ngumi Kuria** — Cybersecurity Analyst (Cyber Shujaa) · Junior Data Scientist (Ngao Labs)

MIT License
