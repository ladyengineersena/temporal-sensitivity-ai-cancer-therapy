"""
Temporal Sensitivity Transformer - time-dependent cancer treatment response modeling.
"""
import torch
import torch.nn as nn


class TemporalSensitivityTransformer(nn.Module):
    """Transformer for temporal sensitivity profiling of cancer treatment response."""

    def __init__(self, feature_dim=3, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim

        self.input_proj = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regression_head = nn.Linear(d_model, 1)

        # Attention layer to extract interpretable attention weights
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, feature_dim)
        Returns:
            preds: (batch,) - predicted temporal sensitivity
            attn_weights: (batch, seq_len) - attention profile for interpretability
        """
        # Project input
        h = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        seq_len = h.shape[1]
        pos = torch.arange(seq_len, device=h.device).float()
        pe = torch.zeros(seq_len, self.d_model, device=h.device)
        for i in range(self.d_model):
            if i % 2 == 0:
                pe[:, i] = torch.sin(pos / (10000 ** (i / self.d_model)))
            else:
                pe[:, i] = torch.cos(pos / (10000 ** ((i - 1) / self.d_model)))
        h = h + pe.unsqueeze(0)

        # Full transformer encoding
        h = self.transformer(h)  # (batch, seq_len, d_model)

        # Attention from last position for interpretability
        query = h[:, -1:, :]  # (batch, 1, d_model)
        key = value = h
        _, attn_weights = self.attn_layer(query, key, value)  # attn_weights: (batch, 1, seq_len)

        # Regression from last position
        h_last = h[:, -1, :]  # (batch, d_model)
        preds = self.regression_head(h_last).squeeze(-1)  # (batch,)

        attn_weights = attn_weights.squeeze(1)  # (batch, seq_len)
        return preds, attn_weights
