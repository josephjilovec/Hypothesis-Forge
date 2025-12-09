"""
Health check API endpoint for production monitoring.
"""
from flask import Flask, jsonify
from utils.health_check import get_health_status

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    status = get_health_status()
    http_status = 200 if status.get("overall_status") == "healthy" else 503
    return jsonify(status), http_status


@app.route("/health/live", methods=["GET"])
def liveness():
    """Liveness probe endpoint."""
    return jsonify({"status": "alive"}), 200


@app.route("/health/ready", methods=["GET"])
def readiness():
    """Readiness probe endpoint."""
    status = get_health_status()
    is_ready = status.get("overall_status") in ("healthy", "warning")
    http_status = 200 if is_ready else 503
    return jsonify({"status": "ready" if is_ready else "not_ready"}), http_status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

