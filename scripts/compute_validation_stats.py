import joblib
import numpy as np
import torch
from pathlib import Path

# Local imports (project root is added via PYTHONPATH when running script)
from src.spaceguard.models.lstm import TelemetryLSTM
from src.spaceguard.data.preprocessing import prepare_channel_data
from src.spaceguard.detection.smoothing import exponential_moving_average


def compute_and_save_stats(chan_id: str = "P-1", z_thresh: float = 3.0) -> None:
    """Compute validation‑set error statistics and persist them on the scaler.

    These statistics are used by the FastAPI service for the calibrated dynamic
    threshold (epsilon = error_mean + z_thresh * error_std).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("data/raw")

    # Prepare data loaders – returns train_loader, val_loader, scaler, in_dim
    _, val_loader, scaler, in_dim = prepare_channel_data(chan_id, data_dir)

    # Load the trained LSTM checkpoint
    ckpt_path = Path("models/checkpoints") / f"{chan_id}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TelemetryLSTM(input_dim=in_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Compute absolute prediction errors on the validation set
    errors = []
    with torch.no_grad():
        for X, y in val_loader:
            preds = model(X.to(device)).squeeze()
            err = torch.abs(preds - y.squeeze()).cpu().numpy()
            errors.append(err)
    errors = np.concatenate(errors)

    # EMA smoothing (window size 10, same as evaluation script)
    smoothed = exponential_moving_average(errors)

    # Store statistics on the scaler object
    scaler.error_mean = float(smoothed.mean())
    scaler.error_std = float(smoothed.std())
    scaler_path = Path("data/processed") / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(
        f"Saved validation error stats to {scaler_path}: "
        f"mean={scaler.error_mean:.6f}, std={scaler.error_std:.6f}"
    )


if __name__ == "__main__":
    compute_and_save_stats()
