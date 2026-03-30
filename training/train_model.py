import os
import sys
import json
import math
import copy
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Project root'tan import için
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_temporal_sensitivity import TemporalSensitivityTransformer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_FEATURES = ["treatment_intensity", "tumor_burden", "toxicity"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sequences(df: pd.DataFrame, features: list[str], seq_len: int = 12):
    """
    Creates sliding windows per patient_id sorted by `time`.

    Returns:
        X: float32 tensor (num_sequences, seq_len, feature_dim)
        y: float32 tensor (num_sequences,)
        patient_ids: numpy array (num_sequences,)
    """
    X, y, patient_ids = [], [], []
    for pid, sub in df.groupby("patient_id"):
        sub = sub.sort_values("time")
        vals = sub[features].to_numpy(dtype=np.float32)
        sens = sub["temporal_sensitivity"].to_numpy(dtype=np.float32)

        for i in range(len(vals) - seq_len):
            X.append(vals[i : i + seq_len])
            y.append(sens[i + seq_len])
            patient_ids.append(pid)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    patient_ids = np.asarray(patient_ids)

    return torch.from_numpy(X), torch.from_numpy(y), patient_ids


def split_patients(patient_ids: np.ndarray, val_ratio: float, test_ratio: float, seed: int):
    unique_pids = np.unique(patient_ids)

    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    # First: split out (val+test) patients from train.
    remaining_ratio = 1.0 - (val_ratio + test_ratio)
    train_ratio_of_total = remaining_ratio
    train_pids, temp_pids = train_test_split(
        unique_pids,
        test_size=(1.0 - train_ratio_of_total),
        random_state=seed,
        shuffle=True,
    )

    # Now split temp into val/test.
    temp_val_fraction = val_ratio / (val_ratio + test_ratio)
    val_pids, test_pids = train_test_split(
        temp_pids,
        test_size=(1.0 - temp_val_fraction),
        random_state=seed,
        shuffle=True,
    )

    return train_pids, val_pids, test_pids


def standardize_from_train(X_train: np.ndarray, X_other: np.ndarray):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return (X_other - mean) / std, mean.squeeze(0), std.squeeze(0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    return {"rmse": float(rmse), "mae": float(mae)}


def run_training(
    data_path: str,
    features: list[str] | None = None,
    seq_len: int = 12,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    device: str = "cpu",
    run_id: str = "latest",
):
    if features is None:
        features = DEFAULT_FEATURES

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}. Run data generator first.")

    set_seed(seed)
    df = pd.read_csv(data_path)

    # Sequences
    X, y, patient_ids = make_sequences(df, features=features, seq_len=seq_len)
    feature_dim = X.shape[-1]

    # Patient-level split to prevent leakage
    train_pids, val_pids, test_pids = split_patients(
        patient_ids=patient_ids, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    train_mask = np.isin(patient_ids, train_pids)
    val_mask = np.isin(patient_ids, val_pids)
    test_mask = np.isin(patient_ids, test_pids)

    X_train = X[train_mask].numpy()
    y_train = y[train_mask].numpy()
    X_val = X[val_mask].numpy()
    y_val = y[val_mask].numpy()
    X_test = X[test_mask].numpy()
    y_test = y[test_mask].numpy()

    # Standardize features using train statistics only
    X_train_2d = X_train.reshape(-1, feature_dim)
    X_train_norm_2d, feature_mean, feature_std = standardize_from_train(X_train_2d, X_train_2d)
    X_val_norm = (X_val.reshape(-1, feature_dim) - feature_mean) / feature_std
    X_test_norm = (X_test.reshape(-1, feature_dim) - feature_mean) / feature_std

    X_train = X_train_norm_2d.reshape(-1, seq_len, feature_dim)
    X_val = X_val_norm.reshape(-1, seq_len, feature_dim)
    X_test = X_test_norm.reshape(-1, seq_len, feature_dim)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )

    use_device = torch.device(device)
    model = TemporalSensitivityTransformer(
        feature_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(use_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(use_device)
            batch_y = batch_y.to(use_device)

            optimizer.zero_grad()
            preds, _ = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        # Validation
        model.eval()
        y_val_preds = []
        y_val_true = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(use_device)
                preds, _ = model(batch_x)
                y_val_preds.append(preds.detach().cpu().numpy())
                y_val_true.append(batch_y.numpy())

        y_val_preds = np.concatenate(y_val_preds, axis=0)
        y_val_true = np.concatenate(y_val_true, axis=0)
        val_metrics = compute_metrics(y_val_true, y_val_preds)

        train_loss_avg = train_loss_total / max(1, len(train_loader))
        print(
            f"Epoch {epoch} | train_mse {train_loss_avg:.6f} | val_rmse {val_metrics['rmse']:.4f} | val_mae {val_metrics['mae']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    y_test_preds = []
    y_test_true = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(use_device)
            preds, _ = model(batch_x)
            y_test_preds.append(preds.detach().cpu().numpy())
            y_test_true.append(batch_y.numpy())

    y_test_preds = np.concatenate(y_test_preds, axis=0)
    y_test_true = np.concatenate(y_test_true, axis=0)
    test_metrics = compute_metrics(y_test_true, y_test_preds)

    # Save artifacts
    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    model_path = os.path.join(project_root, "models", "temporal_sensitivity_model.pt")
    meta_path = os.path.join(project_root, "models", "temporal_sensitivity_model_meta.json")

    torch.save(model.state_dict(), model_path)

    meta = {
        "run_id": run_id,
        "seed": seed,
        "features": features,
        "seq_len": seq_len,
        "model_kwargs": {
            "feature_dim": feature_dim,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
        },
        "normalization": {
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
        },
        "val_rmse_best": best_val_rmse,
        "test_metrics": test_metrics,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Meta saved to {meta_path}")
    print(f"Test metrics: rmse={test_metrics['rmse']:.4f}, mae={test_metrics['mae']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=os.path.join(project_root, "data", "synthetic_temporal_sensitivity.csv"))
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-id", type=str, default="latest")
    args = parser.parse_args()

    run_training(
        data_path=args.data_path,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        device=args.device,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
