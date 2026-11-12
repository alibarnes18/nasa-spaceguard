import torch
import torch.nn as nn


class TelemetryLSTM(nn.Module):
    """NASA SpaceGuard AI - Zaman serisi telemetri tahmini için çok katmanlı LSTM modeli."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 80,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Çok katmanlı LSTM çekirdeği
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Son zaman adımının çıktısını tek bir skaler tahmine dönüştüren lineer katman
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: (batch_size, sequence_length=250, input_dim) boyutunda girdi
            tensörü.

        Returns:
            out: (batch_size, 1) bir sonraki zaman adımının telemetri tahmini.
        """
        # lstm_out: (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Yalnızca son zaman adımının (250. adım) gizli durumunu al
        last_time_step = lstm_out[:, -1, :]

        # Lineer katmandan geçirerek sonraki adımı tahmin et
        prediction = self.linear(last_time_step)
        return prediction