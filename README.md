## Customer Risk Scoring with AutoML

This POC project aims to identify and score risky customers based on behavioural, transactional, and credit-related data. It uses a full pipeline from raw ingestion to automated machine learning (AutoGluon) and model interpretation.

---

### Project Structure

```bash
.
├── dataset/
│   ├── raw/                # Source dataset 
│   ├── bronze/
│   ├── silver/
│   └── gold/               # Final aggregated features
├── notebooks/
│   ├── 00_raw_ingestion.ipynb
│   ├── 01_etl_eda.ipynb
│   ├── 02_agg_summary.ipynb
│   └── 03_model_training_automl.ipynb
├── models/                 # model artifacts and predictions
├── src/
│   ├── data_processing.py
│   ├── data_vis_plotly.py
│   └── feature_engineering.py
├── requirements.txt
└── README.md               
```

---

### Pipeline Overview

1. **Raw Ingestion (`00_raw_ingestion`)**  
   Loads and merges raw Excel files into a unified structure.

2. **ETL & EDA (`01_etl_eda`)**  
   - Cleans and standardises invoice/payment data.
   - Feature engineering.
   - Missing value handling and imputation using credit rating tiers.
   - Exploratory analysis of payment behavior and risk indicators.

3. **Aggregation & Risk Labels (`02_agg_summary`)**  
   - Customer-level metrics: ratios, payment trends, and credit metadata.
   - Constructs gold table used for modeling.
   - Binary label: `target_risky = unpaid_ratio > threshold`.

4. **Model Training with AutoML (`03_model_training_automl`)**  
   - Uses AutoGluon TabularPredictor with stratified train-test split.
   - Evaluates model using ROC AUC, PR curves.
   - Outputs: feature importance, risk probabilities, leaderboard.

---

### Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```
