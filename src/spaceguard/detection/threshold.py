import numpy as np


def compute_dynamic_threshold(
    smoothed_errors: np.ndarray, z_threshold: float = 3.5
) -> float:
    """Yumuşatılmış hata serisinin istatistiklerine dayalı dinamik eşik (epsilon) hesaplar.

    Şartname kuralı: Bu hesaplama asla uzman etiketlerine (ground truth)
    bakmaz!
    """
    mean_err = np.mean(smoothed_errors)
    std_err = np.std(smoothed_errors)

    epsilon = mean_err + (z_threshold * std_err)
    return float(epsilon)


def detect_anomalies_from_errors(
    smoothed_errors: np.ndarray, epsilon: float, offset: int = 250
) -> list[list[int]]:
    """Eşiği aşan hataları tespit eder ve 250 adımlık sliding window ofsetini

    orijinal test zaman indeksine geri taşır.
    """
    mask = smoothed_errors > epsilon
    sequences = []
    in_sequence = False
    start_idx = 0

    for i, val in enumerate(mask):
        if val and not in_sequence:
            in_sequence = True
            start_idx = i
        elif not val and in_sequence:
            in_sequence = False
            sequences.append([start_idx + offset, (i - 1) + offset])

    if in_sequence:
        sequences.append([start_idx + offset, (len(mask) - 1) + offset])

    return sequences
