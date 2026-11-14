"""Unit and integration tests for ticketrouter.serve module."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ticketrouter.serve import app, init_sqlite, state


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["ready"] is True
        assert "uptime_seconds" in data


def test_predict_endpoint_and_sqlite_logging(tmp_path: Path):
    db_file = tmp_path / "test_predictions.db"
    test_conn = init_sqlite(db_file)
    state.db_conn = test_conn

    with TestClient(app) as client:
        # High confidence query
        response = client.post("/predict", json={"text": "How do I activate my new card?"})
        assert response.status_code == 200
        data = response.json()
        assert "intent" in data
        assert "confidence" in data
        assert "route" in data
        assert "latency_ms" in data
        assert isinstance(data["low_confidence"], bool)

        # Verify entry was written to SQLite
        cursor = test_conn.cursor()
        cursor.execute("SELECT prediction, route, confidence FROM predictions")
        rows = cursor.fetchall()
        assert len(rows) >= 1
        assert rows[0][0] == data["intent"]

    # Reset state.db_conn
    state.db_conn = None


def test_prometheus_metrics_endpoint():
    with TestClient(app) as client:
        # Send a prediction request first
        _ = client.post("/predict", json={"text": "Can I cancel my transfer?"})

        # Fetch Prometheus metrics
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        assert "ticketrouter_requests_total" in body
        assert "ticketrouter_latency_seconds" in body
        assert "ticketrouter_confidence_avg" in body
