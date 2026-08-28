"""
Analyst playbooks — recommended next steps for humans.
Aegis never auto-blocks; it informs triage.
"""

from __future__ import annotations

from typing import Any

# High-level guidance only. Not exploit instructions.
PLAYBOOKS: dict[str, dict[str, Any]] = {
    "benign": {
        "summary": "Traffic pattern aligns with baseline benign behavior.",
        "priority": "informational",
        "actions": [
            "No immediate escalation required if context matches known-good assets.",
            "Retain score in case correlation with later alerts is needed.",
            "If this asset is high-value, spot-check against asset inventory.",
        ],
    },
    "probe": {
        "summary": "Pattern consistent with reconnaissance / service discovery.",
        "priority": "medium",
        "actions": [
            "Correlate source with firewall and IDS logs for the same window.",
            "Check whether the source is an approved scanner or unknown external host.",
            "Verify exposure of scanned services; reduce attack surface if unnecessary.",
            "Document in ticket; watch for follow-on exploitation attempts.",
        ],
    },
    "dos": {
        "summary": "Volume / error-rate pattern consistent with denial-of-service style activity.",
        "priority": "high",
        "actions": [
            "Confirm with network utilization and availability metrics.",
            "Identify source aggregates; engage upstream filtering if confirmed abusive.",
            "Protect critical services (rate limits, WAF, Anycast/CDN if applicable).",
            "Preserve flow/logs for post-incident review.",
        ],
    },
    "r2l": {
        "summary": "Pattern consistent with remote access abuse (e.g. repeated auth failures).",
        "priority": "high",
        "actions": [
            "Review authentication logs for the target account/service.",
            "Check lockout policy and MFA status on affected identities.",
            "Block or challenge the source if outside approved ranges.",
            "Reset credentials only through approved identity procedures if compromise suspected.",
        ],
    },
    "u2r": {
        "summary": "Indicators consistent with privilege-elevation style behavior.",
        "priority": "critical",
        "actions": [
            "Treat as high priority: isolate host per IR playbook if confirmed.",
            "Review privileged session logs and recent configuration changes.",
            "Validate integrity of sensitive binaries and auth configuration.",
            "Escalate to incident response; preserve forensic evidence.",
        ],
    },
}


def enrich(result: dict[str, Any]) -> dict[str, Any]:
    """Attach playbook and human-readable rationale to a score result."""
    cls = result.get("predicted_class", "benign")
    pb = PLAYBOOKS.get(cls, PLAYBOOKS["benign"])
    risk = result.get("risk_level", "low")

    rationale = []
    if result.get("anomaly_flag"):
        rationale.append("Unsupervised model flagged this flow as anomalous vs benign baseline.")
    ap = result.get("attack_probability", 0)
    if ap >= 0.5:
        rationale.append(f"Supervised model assigns {ap:.0%} probability of non-benign class.")
    probs = result.get("class_probabilities") or {}
    top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]
    if top:
        rationale.append(
            "Top class probabilities: "
            + ", ".join(f"{k}={v:.0%}" for k, v in top)
        )

    out = dict(result)
    out["analyst"] = {
        "priority": pb["priority"],
        "summary": pb["summary"],
        "recommended_actions": pb["actions"],
        "rationale": rationale,
        "risk_level": risk,
        "disposition": "investigate" if risk in ("high", "critical") or cls != "benign" else "monitor",
    }
    return out
