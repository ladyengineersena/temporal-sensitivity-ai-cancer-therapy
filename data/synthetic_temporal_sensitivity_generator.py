import os
import numpy as np
import pandas as pd

def generate(n_patients=300, timesteps=50, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "synthetic_temporal_sensitivity.csv")

    rows = []

    for pid in range(n_patients):
        tumor = np.random.uniform(0.4, 0.7)
        sensitivity_phase = np.random.uniform(0, 2*np.pi)

        for t in range(timesteps):
            treatment = np.random.uniform(0.3, 1.0)

            # Zamansal duyarlılık dalgası
            temporal_sensitivity = (np.sin(t / 6 + sensitivity_phase) + 1) / 2

            effect = treatment * temporal_sensitivity * 0.08
            toxicity = treatment * (1 - temporal_sensitivity) * 0.05

            tumor += -effect + np.random.normal(0, 0.01)
            tumor = np.clip(tumor, 0, 1)

            rows.append([
                pid, t, treatment,
                temporal_sensitivity,
                tumor, toxicity
            ])

    df = pd.DataFrame(rows, columns=[
        "patient_id",
        "time",
        "treatment_intensity",
        "temporal_sensitivity",
        "tumor_burden",
        "toxicity"
    ])

    df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    generate()
