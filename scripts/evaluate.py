import ast
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from spaceguard.data.preprocessing import (
    TelemetryDataset,
    create_sliding_windows,
    prepare_channel_data,
)
from spaceguard.detection.smoothing import exponential_moving_average
from spaceguard.detection.threshold import (
    compute_dynamic_threshold,
    detect_anomalies_from_errors,
)
from spaceguard.models.lstm import TelemetryLSTM
from spaceguard.models.trainer import predict


def check_overlap(seq_a: list[int], seq_b: list[int]) -> bool:
    """İki zaman aralığının kesişip kesişmediğini denetler."""
    return max(seq_a[0], seq_b[0]) <= min(seq_a[1], seq_b[1])


def evaluate_channel(
    chan_id: str = "P-1", z_thresh: float = 3.0, save_fig: bool = True
):
    """Eğitilmiş LSTM modelini test verisinde değerlendirir, metrikleri hesaplar

    ve açıklayıcı grafikleri kaydeder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("data/raw")
    ckpt_path = Path(f"models/checkpoints/{chan_id}_best.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model ağırlığı bulunamadı: {ckpt_path}. Önce modeli eğitmelisiniz!"
        )

    # 1. Veri boru hattını hazırla
    _, _, scaler, in_dim = prepare_channel_data(chan_id, data_dir)

    # 2. Modeli yükle
    model = TelemetryLSTM(input_dim=in_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Test verisini yükle ve Train Scaler ile ölçekle
    raw_test = np.load(data_dir / "test" / f"{chan_id}.npy")
    scaled_test = raw_test.copy()
    scaled_test[:, 0] = scaler.transform(raw_test[:, [0]]).flatten()

    X_test, y_test = create_sliding_windows(scaled_test, window_size=250)
    test_loader = DataLoader(
        TelemetryDataset(X_test, y_test), batch_size=64, shuffle=False
    )

    # 4. Tahmin üret ve hata serisini hesapla
    y_hat = predict(model, test_loader, device=device)
    raw_errors = np.abs(y_test - y_hat)
    smoothed_errors = exponential_moving_average(raw_errors, window_size=10)

    # 5. Dinamik eşikleme (Unsupervised) ve ofset korumalı tespit
    epsilon = compute_dynamic_threshold(smoothed_errors, z_threshold=z_thresh)
    pred_seqs = detect_anomalies_from_errors(
        smoothed_errors, epsilon, offset=250
    )

    # 6. NASA JPL Event-Level Metriklerini Hesapla
    df_labels = pd.read_csv(data_dir / "labeled_anomalies.csv")
    gt_seqs = ast.literal_eval(
        df_labels[df_labels["chan_id"] == chan_id]["anomaly_sequences"].iloc[0]
    )

    detected_gt = sum(
        1
        for gt in gt_seqs
        if any(check_overlap(gt, pred) for pred in pred_seqs)
    )
    valid_preds = sum(
        1
        for pred in pred_seqs
        if any(check_overlap(pred, gt) for gt in gt_seqs)
    )
    false_alarms = len(pred_seqs) - valid_preds
    recall = detected_gt / len(gt_seqs) if gt_seqs else 0.0
    precision = valid_preds / len(pred_seqs) if pred_seqs else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    print("=" * 60)
    print(f"      NASA SPACEGUARD AI - {chan_id} LSTM DEĞERLENDİRME KARNESİ      ")
    print("=" * 60)
    print(f"Dinamik Eşik (Epsilon): {epsilon:.4f} (z_thresh={z_thresh})")
    print(f"Gerçek Anomali Olayı (Ground Truth): {len(gt_seqs)}")
    print(f"Modelin Tahmin Ettiği Olay Sayısı: {len(pred_seqs)}")
    print(
        f"Yakalanan Gerçek Olay (Recall): {detected_gt}/{len(gt_seqs)} (%{recall*100:.1f})"
    )
    print(f"Yanlış Alarm (False Alarm): {false_alarms}")
    print(f"Hassasiyet (Precision): %{precision*100:.1f}")
    print(f"Event F1-Score: {f1:.4f}")

    # 7. Üç Panelli Profesyonel Açıklayıcı Grafik
    if save_fig:
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        time_index = np.arange(250, len(raw_test))

        # Panel 1: Gerçek Telemetri vs LSTM Tahmini
        axes[0].plot(
            time_index,
            y_test,
            label="Gerçek Telemetri (Actual)",
            color="#1f77b4",
            alpha=0.8,
            linewidth=1.2,
        )
        axes[0].plot(
            time_index,
            y_hat,
            label="LSTM Tahmini (Predicted)",
            color="#ff7f0e",
            linestyle="--",
            alpha=0.8,
            linewidth=1.2,
        )
        for gt in gt_seqs:
            axes[0].axvspan(
                gt[0],
                gt[1],
                color="crimson",
                alpha=0.25,
                label="Ground Truth" if gt == gt_seqs[0] else None,
            )
        axes[0].set_title(
            f"{chan_id} - Gerçek Telemetri vs LSTM Tahmini", fontsize=12
        )
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Yumuşatılmış Tahmin Hatası ve Dinamik Eşik
        axes[1].plot(
            time_index,
            smoothed_errors,
            label="Yumuşatılmış Hata (EMA)",
            color="#2ca02c",
            linewidth=1.2,
        )
        axes[1].axhline(
            epsilon,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Dinamik Eşik (eps={epsilon:.3f})",
        )
        axes[1].set_title(
            "Yumuşatılmış Tahmin Hatası ve Dinamik Eşik", fontsize=12
        )
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Gerçek Anomaliler vs LSTM Alarmları
        for gt in gt_seqs:
            axes[2].axvspan(
                gt[0],
                gt[1],
                color="crimson",
                alpha=0.3,
                label="Gerçek Anomali" if gt == gt_seqs[0] else None,
            )
        for pred in pred_seqs:
            axes[2].axvspan(
                pred[0],
                pred[1],
                color="blue",
                alpha=0.3,
                label="LSTM Tespiti" if pred == pred_seqs[0] else None,
            )
        axes[2].set_title(
            "Gerçek Anomaliler (Kırmızı) vs LSTM Alarmları (Mavi)", fontsize=12
        )
        axes[2].set_xlabel("Zaman Adımı (Time Step)", fontsize=11)
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = Path(f"reports/figures/{chan_id}_lstm_detection.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150)
        print(f"\n[Açıklayıcı Grafik Kaydedildi]: {fig_path}")

    return {
        "recall": recall,
        "precision": precision,
        "false_alarms": false_alarms,
        "f1": f1,
    }


if __name__ == "__main__":
    evaluate_channel("P-1", z_thresh=3.0)

