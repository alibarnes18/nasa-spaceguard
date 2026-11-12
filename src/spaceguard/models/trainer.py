from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    """Deneylerin tekrarlanabilirliği (reproducibility) için tüm rastgelelik tohumlarını sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 35,
    lr: float = 0.001,
    patience: int = 5,
    device: torch.device = None,
    checkpoint_path: Path = None,
) -> dict[str, list[float]]:
    """Early stopping ve checkpoint destekli PyTorch LSTM model eğitim döngüsü."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Eğitim Başladı] Donanım: {device} | Maksimum Epoch: {epochs} | Early Stopping Patience: {patience}"
    )

    for epoch in range(1, epochs + 1):
        # 1. Eğitim Aşaması
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(X_batch)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # 2. Doğrulama (Validation) Aşaması
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                running_val_loss += loss.item() * len(X_batch)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] -> Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}"
        )

        # 3. Early Stopping ve Checkpoint Kontrolü
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            if checkpoint_path is not None:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                        "input_dim": getattr(model, "input_dim", None),
                    },
                    checkpoint_path,
                )
                print(f"   -> En iyi model ağırlığı kaydedildi: {checkpoint_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\n[Early Stopping] Doğrulama kaybı {patience} epoch boyunca iyileşmedi. Eğitim erken sonlandırıldı!"
                )
                break

    # En iyi ağırlıkları modele geri yükle
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n[Tamamlandı] En iyi model yüklendi (En düşük Val Loss: {best_val_loss:.6f})")

    return history

def predict(
    model: nn.Module, data_loader: DataLoader, device: torch.device = None
) -> np.ndarray:
    """Eğitilmiş model ile veri seti üzerinde sonraki adım tahminlerini üretir."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            predictions.append(preds.cpu().numpy())

    return np.vstack(predictions)