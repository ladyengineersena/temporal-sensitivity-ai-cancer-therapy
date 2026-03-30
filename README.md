# Temporal Sensitivity AI in Cancer Therapy
This project models time-dependent sensitivity patterns of cancer
treatment response using Transformer-based temporal modeling.

## Environment
This project is a research prototype.
Environment dependencies are defined via `requirements.txt`.
Dockerization is intentionally omitted.

## Key Contributions
- Temporal sensitivity profiling
- Transformer with attention-based interpretability
- Synthetic tumor response simulation
- Ethical, non-clinical research design

## Quick Start
```bash
pip install -r requirements.txt
python run_all.py --seed 42
```

This runs: data generation → training → analysis. Outputs:
- `models/temporal_sensitivity_model.pt` – trained model (created locally)
- `models/temporal_sensitivity_model_meta.json` – feature scaling params + metrics (RMSE/MAE, created locally)
- `analysis/outputs/sensitivity_profile.png` – temporal sensitivity profile plot (created locally)

Or run steps individually:
- `python data/synthetic_temporal_sensitivity_generator.py`
- `python training/train_model.py`
- `python analysis/sensitivity_profile_analysis.py`

Training now uses:
- Patient-level split to reduce leakage risk
- Seeded reproducibility
- Feature standardization (train statistics only)
- RMSE/MAE evaluation on validation and test sets

## Tests
```bash
pytest -q
```

## Disclaimer
This repository does not provide medical advice.

## License
NO LICENSE – All rights reserved.
