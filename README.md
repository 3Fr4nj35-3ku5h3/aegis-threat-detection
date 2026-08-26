# Aegis — AI-Assisted Threat Detection

**Hybrid network-flow threat detection:** unsupervised anomaly scoring + supervised attack classification, with a live dashboard and REST API.

Built as a **portfolio-grade, reproducible** security ML system — transparent metrics, honest scope, runnable in minutes.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-22d3ee?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-34d399?style=for-the-badge)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

---

## Why Aegis is different

| Principle | How Aegis does it |
|-----------|-------------------|
| **Hybrid detection** | Isolation Forest (anomaly) + Random Forest (benign / probe / DoS / R2L / U2R) |
| **Fused risk score** | Combines attack probability + anomaly signal → low / medium / high / critical |
| **Reproducible** | Synthetic KDD-style flows generated in-repo — no opaque data dumps |
| **Transparent** | Held-out ROC-AUC, PR-AUC, macro-F1, confusion matrix, feature importances |
| **Demo-ready** | FastAPI + dark SOC-style dashboard with curated attack scenarios |
| **Honest scope** | Flow-feature classifier for education & portfolio — not a full NIDS product |

---

## Architecture

```
Network-flow features (30 dims)
            │
            ▼
     StandardScaler
        ┌───┴───┐
        ▼       ▼
 Isolation   Random Forest
  Forest     (5-class)
        └───┬───┘
            ▼
   Fused risk score + labels
            │
     FastAPI  +  Dashboard
```

**Attack taxonomy (demo labels)**

| ID | Class | Meaning (simplified) |
|----|-------|----------------------|
| 0 | benign | Normal traffic |
| 1 | probe | Scanning / reconnaissance |
| 2 | dos | Denial-of-service style volume |
| 3 | r2l | Remote-to-local (e.g. brute force patterns) |
| 4 | u2r | User-to-root / privilege indicators |

---

## Quick start

```bash
git clone https://github.com/3Fr4nj35-3ku5h3/aegis-threat-detection.git
cd aegis-threat-detection

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1) Generate synthetic training data
python src/generate_data.py --n 12000

# 2) Train models + write metrics
python src/train.py

# 3) Launch API + dashboard
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000** — select a scenario → **Score selected**.

### API examples

```bash
curl -s http://127.0.0.1:8000/api/health | jq
curl -s http://127.0.0.1:8000/api/metrics | jq
curl -s -X POST http://127.0.0.1:8000/api/score/one \
  -H 'Content-Type: application/json' \
  -d '{"flow":{"duration":0.2,"protocol_type":0,"src_bytes":20,"count":120,"same_srv_rate":0.1}}' | jq
```

---

## Project layout

```
aegis-threat-detection/
├── src/
│   ├── generate_data.py   # synthetic flow generator
│   ├── train.py           # IsolationForest + RandomForest
│   ├── infer.py           # scoring engine
│   ├── api.py             # FastAPI app
│   └── features.py        # schema + label maps
├── static/index.html      # detection dashboard
├── data/                  # flows.csv (generated)
├── models/                # joblib artifacts (generated)
├── reports/metrics.json
└── requirements.txt
```

---

## Typical metrics (synthetic hold-out)

After `python src/train.py` you should see strong separation on this controlled dataset, for example:

- Isolation Forest ROC-AUC (attack vs benign) ≈ **0.90+**
- Random Forest ROC-AUC (attack vs benign) ≈ **0.95+**
- Macro-F1 across 5 classes ≈ **0.90+**

Exact numbers are written to `reports/metrics.json` on every train run — **cite those**, not this README.

---

## Security & ethics

- For **authorized research, education, and portfolio demonstration** only  
- Synthetic features — not a license to scan networks you do not own  
- Models can be wrong; never auto-block production traffic from a demo classifier  
- Unauthorized access to systems is illegal  

---

## Author

**Francis Ngumi Kuria**  
Cybersecurity Analyst (Cyber Shujaa — Security Analyst With Pass)  
Junior Data Scientist (Ngao Labs — Foundations of DS & AI)  

[Portfolio](https://3fr4nj35-3ku5h3.github.io/francis-kuria.github.io/) · [GitHub](https://github.com/3Fr4nj35-3ku5h3)

MIT License
