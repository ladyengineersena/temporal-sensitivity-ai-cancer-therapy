#!/usr/bin/env python
"""
Temporal Sensitivity AI - Full pipeline: data generation, training, analysis.
"""
import os
import sys

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    print("=" * 50)
    print("1. Generating synthetic data...")
    print("=" * 50)
    from data.synthetic_temporal_sensitivity_generator import generate
    generate()
    print("Done.\n")

    print("=" * 50)
    print("2. Training model...")
    print("=" * 50)
    from training.train_model import run_training
    run_training()
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
