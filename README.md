
# CVF-Net — Cross-View Fusion for Intelligent Traffic Congestion Analysis
End-to-end skeleton implementing the paper pipeline with runnable forecasting on METR-LA/PEMS-BAY (synthetic fallback), ablations, explainability, and robustness.
## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
## Quick Start
```bash
python scripts/train_forecasting.py --dataset metr-la --config configs/forecasting_base.yaml
python scripts/train_forecasting.py --dataset pems-bay --config configs/forecasting_base.yaml
python scripts/run_ablation.py --dataset metr-la --config configs/forecasting_base.yaml
python scripts/run_explainability.py --dataset metr-la --config configs/forecasting_base.yaml
python scripts/run_robustness.py --dataset metr-la --config configs/forecasting_base.yaml
```
Data paths under `data/`. If missing, synthetic data is generated.
