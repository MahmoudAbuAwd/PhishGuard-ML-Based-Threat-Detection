# PhishGuard — ML-Based Threat Detection

## Overview
This repository detects phishing websites using classical ML models trained on URL-derived features. It replaces the notebook workflow with reproducible Python scripts and a simple pipeline.

## Project Structure
- `data/` — raw CSV dataset (`phishing.csv`)
- `models/` — saved models (`best_model.pkl`)
- `src/`
  - `data_processing.py` — dataset loading and preprocessing
  - `phishing_pipeline.py` — training, evaluation, and model selection
- `scripts/`
  - `preprocess.py` — generate processed dataset with selected features
- `train.py` — run training and save the best model
- `evaluate.py` — evaluate a saved model
- `requirements.txt` — Python dependencies
- `.gitignore` — ignore build artifacts

## Setup
1. Create a virtual environment and install dependencies:
   - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
   - Install: `pip install -r requirements.txt`

2. Ensure dataset is present at `data/phishing.csv`.

## Preprocessing (optional)
Generate a processed dataset (`data/processed.csv`) containing the selected features and labels:
- `python scripts/preprocess.py --input data/phishing.csv --output data/processed.csv`

## Training
Train multiple models, select the best by F1 score, and save to `models/best_model.pkl`:
- `python train.py --data-path data/phishing.csv --model-path models/best_model.pkl`

## Evaluation
Evaluate the saved model on the standard 20% test split:
- `python evaluate.py --data-path data/phishing.csv --model-path models/best_model.pkl`

## Notes
- Notebook is not used for training; the Python pipeline fully replaces it.
- Default selected features: `PrefixSuffix-`, `SubDomains`, `HTTPS`, `AnchorURL`, `WebsiteTraffic`.

