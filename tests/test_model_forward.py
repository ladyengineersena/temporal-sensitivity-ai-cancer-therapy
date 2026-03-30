import torch

from models.transformer_temporal_sensitivity import TemporalSensitivityTransformer


def test_forward_shapes():
    model = TemporalSensitivityTransformer(
        feature_dim=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.0,
    )
    x = torch.randn(2, 12, 3)

    preds, attn = model(x)

    assert preds.shape == (2,)
    assert attn.shape == (2, 12)
    assert torch.isfinite(preds).all()
    assert torch.isfinite(attn).all()

