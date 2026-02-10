# Temporal Sensitivity AI in Cancer Therapy

This project models time-dependent sensitivity patterns of cancer
treatment response using Transformer-based temporal modeling.

## Environment
This project is a research prototype.
Environment dependencies are defined via requirements.txt.
Dockerization is intentionally omitted.


## Key Contributions
- Temporal sensitivity profiling
- Transformer with attention-based interpretability
- Synthetic tumor response simulation
- Ethical, non-clinical research design

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py
```

This runs: data generation → training → analysis. Outputs:
- `models/temporal_sensitivity_model.pt` – trained model
- `analysis/outputs/sensitivity_profile.png` – temporal sensitivity profile plot

Or run steps individually:
- `python data/synthetic_temporal_sensitivity_generator.py`
- `python training/train_model.py`
- `python analysis/sensitivity_profile_analysis.py`

## Disclaimer
This repository does not provide medical advice.

## License
NO LICENSE – All rights reserved.

