import os
import sys
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

    if not os.path.exists(csv_path):
        raise FileNotFoundError("Run data generator first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run training first.")

    df = pd.read_csv(csv_path)
    features = ["treatment_intensity", "tumor_burden", "toxicity"]

    pid = df.patient_id.unique()[0]
    sub = df[df.patient_id == pid][features].values[:12]
    x = torch.tensor(sub).float().unsqueeze(0)

    model = TemporalSensitivityTransformer(feature_dim=3)
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
