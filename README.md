# Housing-Prices-Predictor
Predict movement of Toronto condo rentals over the next year.

## Table of contents
- **Overview**: What this project does
- **Data**: Source & processing scripts
- **Usage**: Setup and run instructions
- **Project structure**: Important files
- **Next steps**: Suggested improvements

## Overview
This project aims to predict short-term trends in Toronto condominium rental prices using historical rent time series. It includes scripts to prepare the dataset and a simple regression experiment to model rental price movement.

## Data
- **Raw data**: [data/raw/historical_rent_timeseries.csv](data/raw/historical_rent_timeseries.csv)
- **Processed data**: [data/processed/historical_rent_timeseries.csv](data/processed/historical_rent_timeseries.csv)

Data preparation is handled by the script: [data/create_dataset.py](data/create_dataset.py).

## Quick start (Windows)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies (recommended):

Option A — install from `requirements.txt` (recommended):

```powershell
pip install -r requirements.txt
```

Option B — install core packages manually:

```powershell
pip install pandas numpy scikit-learn matplotlib
```

3. Prepare the dataset:

```powershell
python data/create_dataset.py
```

4. Run the regression experiment:

```powershell
python Regression/regression.py
```

## Project structure
- **data/**: dataset creation and raw/processed CSVs
	- `create_dataset.py` — load and transform raw time-series into model-ready CSV
- **Regression/**: modeling code
	- `regression.py` — example regression pipeline and evaluation
- **README.md** — this file
 - `requirements.txt` — project dependencies for easy install

## Next steps & ideas
- Add `requirements.txt` or `pyproject.toml` for reproducible installs.
- Improve modeling (feature engineering, cross-validation, model tracking).
- Add visualizations and example notebooks for EDA.

## License & Contact
This repository is an educational project. For questions or collaboration, open an issue or contact the repo owner.

