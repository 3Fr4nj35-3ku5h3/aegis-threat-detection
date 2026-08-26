"""Feature schema and helpers for network-flow style records."""

from __future__ import annotations

FEATURE_COLUMNS = [
    "duration",
    "protocol_type",  # 0=tcp, 1=udp, 2=icmp
    "src_bytes",
    "dst_bytes",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_serror_rate",
    "dst_host_rerror_rate",
]

# Human-readable labels for dashboard
LABEL_MAP = {
    0: "benign",
    1: "probe",
    2: "dos",
    3: "r2l",
    4: "u2r",
}

PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}
