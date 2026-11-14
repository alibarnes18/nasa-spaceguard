import sys
import pathlib
import json

# Add project root to sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import health_check, predict, TelemetryWindow

def load_payload():
    payload_path = ROOT_DIR / "payload.json"
    with open(payload_path, "r") as f:
        return json.load(f)

def test_health():
    res = health_check()
    assert res.get("status") == "healthy"
    assert res.get("model_loaded") is True

def test_predict():
    payload = load_payload()
    window = TelemetryWindow(data=payload["data"])
    res = predict(window)
    for key in ["prediction", "true_value", "error", "threshold", "is_anomaly", "z_thresh"]:
        assert key in res, f"Missing key {key}"
    assert isinstance(res["prediction"], float)
    assert isinstance(res["is_anomaly"], bool)

if __name__ == "__main__":
    test_health()
    test_predict()
    print("All API tests passed successfully!")

