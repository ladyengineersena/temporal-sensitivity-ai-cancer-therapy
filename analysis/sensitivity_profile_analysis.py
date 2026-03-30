import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_temporal_sensitivity import TemporalSensitivityTransformer

def plot_sensitivity(attn, save_path=None):
    weights = attn.squeeze().detach().numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(weights)
    plt.xlabel("Time")
    plt.ylabel("Attention Weight")
    plt.title("Temporal Sensitivity Profile")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def run_analysis():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "synthetic_temporal_sensitivity.csv")
    model_path = os.path.join(project_root, "models", "temporal_sensitivity_model.pt")
    meta_path = os.path.join(project_root, "models", "temporal_sensitivity_model_meta.json")

    if not os.path.exists(csv_path):
        raise FileNotFoundError("Run data generator first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run training first.")

    df = pd.read_csv(csv_path)
    features = ["treatment_intensity", "tumor_burden", "toxicity"]
    seq_len = 12
    model_kwargs = {"d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.1}
    feature_mean = None
    feature_std = None

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        features = meta.get("features", features)
        seq_len = int(meta.get("seq_len", seq_len))
        model_kwargs = meta.get("model_kwargs", model_kwargs)
        norm = meta.get("normalization", {})
        feature_mean = norm.get("feature_mean", None)
        feature_std = norm.get("feature_std", None)

    pid = df.patient_id.unique()[0]
    sub = df[df.patient_id == pid][features].values[:12]
    sub = df[df.patient_id == pid].sort_values("time")[features].values[:seq_len]
    x = torch.tensor(sub).float().unsqueeze(0)  # (1, seq_len, feature_dim)

    # Apply the same standardization used during training (if available)
    if feature_mean is not None and feature_std is not None:
        mean_t = torch.tensor(feature_mean, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        std_t = torch.tensor(feature_std, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = (x - mean_t) / std_t

    model = TemporalSensitivityTransformer(
        feature_dim=len(features),
        d_model=int(model_kwargs.get("d_model", 64)),
        nhead=int(model_kwargs.get("nhead", 4)),
        num_layers=int(model_kwargs.get("num_layers", 2)),
        dropout=float(model_kwargs.get("dropout", 0.1)),
    )
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        _, attn = model(x)

    out_dir = os.path.join(project_root, "analysis", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    plot_sensitivity(attn, save_path=os.path.join(out_dir, "sensitivity_profile.png"))
    print("Plot saved to analysis/outputs/sensitivity_profile.png")


if __name__ == "__main__":
    run_analysis()
