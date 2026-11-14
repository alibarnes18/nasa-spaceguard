# dashboard/app.py
import streamlit as st
import httpx
import json
import pathlib

API_URL = "http://127.0.0.1:8000/api/predict"

@st.cache_data
def load_payload():
    p = pathlib.Path(__file__).parent.parent / "payload.json"
    return json.loads(p.read_text())

def predict():
    payload = load_payload()
    r = httpx.post(API_URL, json=payload, timeout=10)
    return r.json()

st.title("🚀 NASA SpaceGuard – Anomaly Detection")
st.write("**Payload** (250 × in_dim) gönderiliyor ve model tahmini alınıyor.")
if st.button("Tahmin Et"):
    result = predict()
    st.json(result)
    # Görsel iyileştirme (isteğe bağlı)
    if result["is_anomaly"]:
        st.error("⚠️ Anomali tespit edildi!")
    else:
        st.success("✅ Anomali yok.")