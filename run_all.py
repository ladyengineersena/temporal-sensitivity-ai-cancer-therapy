#!/usr/bin/env python
"""
Temporal Sensitivity AI - Full pipeline: data generation, training, analysis.
"""
import os
import sys
import argparse

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-patients", type=int, default=300)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-id", type=str, default="latest")
    args = parser.parse_args()

    print("=" * 50)
    print("1. Generating synthetic data...")
    print("=" * 50)
    from data.synthetic_temporal_sensitivity_generator import generate
    generate(
        n_patients=args.n_patients,
        timesteps=args.timesteps,
        seed=args.seed,
    )
    print("Done.\n")

    print("=" * 50)
    print("2. Training model...")
    print("=" * 50)
    from training.train_model import run_training
    run_training(
        data_path=os.path.join(project_root, "data", "synthetic_temporal_sensitivity.csv"),
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        run_id=args.run_id,
    )
    print("Done.\n")

    print("=" * 50)
    print("3. Running analysis...")
    print("=" * 50)
    from analysis.sensitivity_profile_analysis import run_analysis
    run_analysis()
    print("Done.\n")

    print("=" * 50)
    print("Pipeline complete!")
    print("  - Model: models/temporal_sensitivity_model.pt")
    print("  - Plot:  analysis/outputs/sensitivity_profile.png")
    print("=" * 50)


if __name__ == "__main__":
    main()
