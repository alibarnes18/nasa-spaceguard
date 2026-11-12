from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset


def create_sliding_windows(
    data: np.ndarray, window_size: int = 250, target_col: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Zaman serisi matrisini kayan pencere (sliding window) formatına dönüştürür.

    Args:
        data: (T, D) boyutunda zaman serisi dizisi.
        window_size: Girdi penceresi uzunluğu (NASA standardı: 250).
        target_col: Tahmin edilecek hedef sütun indeksi (0: telemetri).

    Returns:
        X: (N, window_size, D) girdi tensörü
        y: (N, 1) hedef tensörü (bir sonraki zaman adımının telemetri değeri)
    """
    num_samples = len(data) - window_size
    if num_samples <= 0:
        raise ValueError(
            f"Veri uzunluğu ({len(data)}) pencere boyutundan ({window_size}) büyük olmalıdır!"
        )

    X = np.empty((num_samples, window_size, data.shape[1]), dtype=data.dtype)
    y = np.empty((num_samples, 1), dtype=data.dtype)

    for i in range(num_samples):
        X[i] = data[i : i + window_size]
        y[i] = data[i + window_size, target_col]

    return X, y


class TelemetryDataset(Dataset):
    """PyTorch için Tensör Veri Seti Sarmalayıcısı."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def prepare_channel_data(
    chan_id: str,
    data_dir: Path,
    window_size: int = 250,
    val_ratio: float = 0.2,
    batch_size: int = 64,
):
    """Bir kanalın train verisini okur, ölçekler, kronolojik olarak böler ve

    PyTorch DataLoader nesnelerini hazırlar.
    """
    raw_train = np.load(data_dir / "train" / f"{chan_id}.npy")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_train = raw_train.copy()
    scaled_train[:, 0] = scaler.fit_transform(raw_train[:, [0]]).flatten()

    X, y = create_sliding_windows(scaled_train, window_size=window_size)

    split_idx = int(len(X) * (1 - val_ratio))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    train_dataset = TelemetryDataset(X_train, y_train)
    val_dataset = TelemetryDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler, raw_train.shape[1]