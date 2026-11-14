# create_payload.py — güncellenmiş
import numpy as np, json
from pathlib import Path

DATA_DIR = Path("data/raw")
CHAN_ID = "P-1"
WINDOW_SIZE = 250

raw_test = np.load(DATA_DIR / "test" / f"{CHAN_ID}.npy")

# İlk 250 adımı al (eğer eksikse, sonundan kopar)
window = raw_test[:WINDOW_SIZE]

# Eğer veri <250 adım ise hata ver
if window.shape[0] < WINDOW_SIZE:
    raise RuntimeError(f"Test setinde yeterli veri yok (found {window.shape[0]} < {WINDOW_SIZE})")

# Gerekirse eksik sütunları 0 ile doldur (25 sütun garantili)
if window.shape[1] != 25:
    raise RuntimeError(f"Beklenen 25 özellik, ama {window.shape[1]} bulundu.")

payload = {"data": window.tolist()}
Path("payload.json").write_text(json.dumps(payload, indent=2))
print("payload.json oluşturuldu →", window.shape)