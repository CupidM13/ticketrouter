"""Streamlit Monitoring Dashboard for TicketRouter."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path so 'ticketrouter' can be imported anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ticketrouter.monitor import (  # noqa: E402
    evaluate_drift_and_quality,
    get_training_intent_distribution,
)

# Page Configuration
st.set_page_config(
    page_title="TicketRouter MLOps Monitor",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

SQLITE_DB_PATH = Path(os.environ.get("SQLITE_DB_PATH", "predictions.db"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RETRAIN_REQUEST_FILE = Path("retrain_request.json")
ALERT_HISTORY_FILE = Path("metrics/alert_history.json")


def load_recent_predictions(limit: int = 500) -> pd.DataFrame:
    """Load latest predictions from SQLite."""
    if not SQLITE_DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(SQLITE_DB_PATH)
    query = f"""
        SELECT id, timestamp, text_hash, prediction, confidence, route, latency_ms, low_confidence
        FROM predictions
        ORDER BY id DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_production_model_info() -> dict[str, str]:
    """Retrieve production model alias and version info."""
    candidate_metrics_file = Path("metrics/candidate_metrics.json")
    if candidate_metrics_file.exists():
        try:
            with open(candidate_metrics_file, encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "model_name": data.get("model_name", "ticketrouter_model"),
                    "version": "1 (@Production)",
                    "status": "Healthy / Serving",
                    "base_model": os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
                }
        except Exception:
            pass
    return {
        "model_name": "ticketrouter_model",
        "version": "1 (@Production)",
        "status": "Healthy / Serving",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    }


# Header
st.title("🧭 TicketRouter: Intent Routing & MLOps Monitor")
model_info = get_production_model_info()

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Production Model", model_info["model_name"])
with col_m2:
    st.metric("Registry Version", model_info["version"])
with col_m3:
    st.metric("Base Architecture", model_info["base_model"].split("/")[-1])
with col_m4:
    st.metric("Service Status", "🟢 Healthy")

st.divider()

# Sidebar Controls
st.sidebar.header("⚙️ Monitoring Controls")
window_size = st.sidebar.slider(
    "Rolling Window Size", min_value=50, max_value=500, value=150, step=25
)
auto_refresh = st.sidebar.checkbox("Auto Refresh (every 5s)", value=False)
if auto_refresh:
    time.sleep(5)
    st.rerun()

if st.sidebar.button("🔄 Refresh Data Now"):
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("🚨 Alert Thresholds")
psi_thresh = st.sidebar.number_input("PSI Drift Threshold", value=0.20, step=0.05)
fallback_thresh = st.sidebar.number_input("Fallback Rate Threshold", value=0.35, step=0.05)
conf_drop_thresh = st.sidebar.number_input("Confidence Drop Threshold", value=0.15, step=0.05)

# Load Data
df = load_recent_predictions(limit=window_size)

if df.empty:
    st.info(
        "No prediction logs found in SQLite yet. "
        "Run `python -m ticketrouter.monitor --replay` or send queries to `/predict` to generate traffic."
    )
    if st.button("🚀 Replay Live Traffic Simulation"):
        with st.spinner("Replaying traffic (clean then shifted)..."):
            from ticketrouter.monitor import replay_traffic

            replay_traffic(num_samples_per_split=window_size)
        st.success("Replay complete! Refreshing page...")
        st.rerun()
    st.stop()

# Evaluate Drift Metrics
drift_res = evaluate_drift_and_quality(
    db_path=SQLITE_DB_PATH,
    data_dir=DATA_DIR,
    window_size=window_size,
    psi_threshold=psi_thresh,
    fallback_threshold=fallback_thresh,
    confidence_drop_threshold=conf_drop_thresh,
)

m = drift_res.get("metrics", {})
psi_val = m.get("psi", 0.0)
mean_conf = m.get("mean_confidence", 0.0)
fallback_rate = m.get("fallback_rate", 0.0)
p95_lat = m.get("latency_p95_ms", 0.0)

# High-Level Metrics Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Window Requests", len(df))
with kpi2:
    psi_delta = (
        "🔴 DRIFT" if psi_val >= psi_thresh else ("🟡 SLIGHT" if psi_val >= 0.10 else "🟢 STABLE")
    )
    st.metric(
        "PSI Drift Index",
        f"{psi_val:.4f}",
        delta=psi_delta,
        delta_color="inverse" if psi_val >= psi_thresh else "normal",
    )
with kpi3:
    fb_delta = "🔴 HIGH" if fallback_rate >= fallback_thresh else "🟢 NORMAL"
    st.metric(
        "Fallback Rate",
        f"{fallback_rate * 100:.1f}%",
        delta=fb_delta,
        delta_color="inverse" if fallback_rate >= fallback_thresh else "normal",
    )
with kpi4:
    st.metric("Mean Confidence", f"{mean_conf:.3f}")
with kpi5:
    st.metric("Latency (p95)", f"{p95_lat:.1f} ms")

# Active Retrain Trigger Banner
if RETRAIN_REQUEST_FILE.exists():
    try:
        with open(RETRAIN_REQUEST_FILE, encoding="utf-8") as f:
            retrain_data = json.load(f)
            if retrain_data.get("alert_triggered"):
                st.error("🚨 **RETRAIN TRIGGER ACTIVE**: Drift or Quality degradation detected!")
                for reason in retrain_data.get("reasons", []):
                    st.write(f"- ⚠️ {reason}")
                c_btn1, c_btn2 = st.columns([1, 4])
                with c_btn1:
                    if st.button("✅ Acknowledge & Clear Alert"):
                        RETRAIN_REQUEST_FILE.unlink(missing_ok=True)
                        st.success("Alert cleared.")
                        st.rerun()
                with c_btn2:
                    st.caption(
                        "A `retrain_request.json` file has been written to trigger automated CI/CD retraining."
                    )
    except Exception:
        pass

st.divider()

# Charts
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Distribution & Drift Analysis",
        "🛣️ Route Breakdown & Confidence",
        "📜 Live Logs & Alert History",
    ]
)

with tab1:
    col_chart1, col_chart2 = st.columns(2)
    train_counts, all_labels = get_training_intent_distribution(DATA_DIR)
    live_counts = df["prediction"].value_counts().to_dict()

    # Top 15 Intents Bar Comparison
    top_intents = sorted(train_counts.keys(), key=lambda k: train_counts[k], reverse=True)[:15]
    total_train = sum(train_counts.values())
    total_live = max(1, sum(live_counts.values()))

    chart_df = pd.DataFrame(
        {
            "Training Baseline (%)": [
                (train_counts.get(i, 0) / total_train) * 100 for i in top_intents
            ],
            "Live Traffic Window (%)": [
                (live_counts.get(i, 0) / total_live) * 100 for i in top_intents
            ],
        },
        index=top_intents,
    )

    with col_chart1:
        st.subheader("Intent Distribution: Baseline vs Live Traffic")
        st.bar_chart(chart_df)

    with col_chart2:
        st.subheader("Confidence Score Distribution")
        st.area_chart(df["confidence"].reset_index(drop=True))

with tab2:
    col_rt1, col_rt2 = st.columns(2)
    with col_rt1:
        st.subheader("Routing Distribution")
        route_counts = df["route"].value_counts()
        st.bar_chart(route_counts)

    with col_rt2:
        st.subheader("Per-Request Latency (ms)")
        st.line_chart(df["latency_ms"].reset_index(drop=True))

with tab3:
    st.subheader("Recent Request Logs (Last 50)")
    try:
        st.dataframe(
            df[
                [
                    "id",
                    "timestamp",
                    "prediction",
                    "confidence",
                    "route",
                    "latency_ms",
                    "low_confidence",
                ]
            ].head(50),
            width="stretch",
        )
    except Exception:
        st.dataframe(
            df[
                [
                    "id",
                    "timestamp",
                    "prediction",
                    "confidence",
                    "route",
                    "latency_ms",
                    "low_confidence",
                ]
            ].head(50),
            use_container_width=True,
        )

    if ALERT_HISTORY_FILE.exists():
        st.subheader("Alert History")
        try:
            with open(ALERT_HISTORY_FILE, encoding="utf-8") as f:
                history = json.load(f)
            st.json(history[-5:])
        except Exception:
            pass
