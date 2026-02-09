import os
import sys
import numpy as np
import torch
import pandas as pd

# Project root'tan import için
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_temporal_sensitivity import TemporalSensitivityTransformer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, "data", "synthetic_temporal_sensitivity.csv")

features = [
    "treatment_intensity",
    "tumor_burden",
    "toxicity"
]


def make_sequences(df, seq_len=12):
    X, y = [], []
    for pid in df.patient_id.unique():
        sub = df[df.patient_id == pid]
        vals = sub[features].values
        sens = sub["temporal_sensitivity"].values
        for i in range(len(vals) - seq_len):
            X.append(vals[i:i+seq_len])
            y.append(sens[i+seq_len])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


def run_training():
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data not found: {csv_path}. Run data generator first.")

    df = pd.read_csv(csv_path)
    X, y = make_sequences(df)

    model = TemporalSensitivityTransformer(feature_dim=X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    BATCH_SIZE = 256
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(10):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds, _ = model(batch_x)
            loss = loss_fn(preds.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} | Loss {total_loss / len(loader):.4f}")

    model_path = os.path.join(project_root, "models", "temporal_sensitivity_model.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    run_training()
