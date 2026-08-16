# Ticket Sales Forecasting for Piccolo Teatro di Milano

An end-to-end machine-learning pipeline that forecasts the cumulative ticket-sales curve of theatre shows. Developed for **Piccolo Teatro di Milano** as a thesis project, it is designed to help the marketing team identify shows that may benefit from additional advertising investment.

The system turns raw box-office transactions, show metadata, and season calendars into daily show-level features, then produces iterative multi-day forecasts with an XGBoost bootstrap ensemble and uncertainty bounds suitable for dashboarding.

## Highlights

- Built a reproducible data pipeline for 553k+ box-office transactions, performance metadata, and theatre-season calendars.
- Engineered cumulative sales, rolling-average, lag, change, capacity, and sales-calendar features at show/day granularity.
- Trained a global XGBoost model that generalizes to unseen shows, using show-level splits and `GroupKFold` to prevent leakage across a show's observations.
- Implemented iterative forecasting: each daily prediction updates the feature state used to forecast the following day.
- Estimated 90% uncertainty intervals with a 70-model bootstrap ensemble; model hyperparameters are selected through Bayesian optimization.
- Prepared forecasts for consumption in a Power BI marketing dashboard.

## Business problem

Ticket sales begin at the start of a theatre season and continue until each show’s final performance. Marketing teams need a reliable view of the expected sales trajectory early enough to decide whether to increase promotion.

Rather than fitting one time-series model per show, this project learns a single supervised model from historical shows and forecasts the **cumulative percentage of tickets sold** for new ones. Normalizing by capacity makes sales curves comparable across productions of different sizes.

## Approach

```text
Raw transactions + show catalogue + season calendar
                       |
                 Cleaning & validation
                       |
       Daily, show-level feature engineering
                       |
     Show-level train / validation / test split
                       |
Bayesian-tuned XGBoost bootstrap ensemble (70 models)
                       |
  Iterative sales-curve forecast + 90% interval
                       |
          Power BI-ready predictions and metrics
```

### Feature engineering

The model uses sales history and the position of each show in its selling window, including:

- cumulative revenue and ticket sales;
- cumulative sell-through percentage (the log-transformed prediction target);
- rolling averages, lagged values, and deltas over 2–30 day windows;
- remaining capacity, days since sales opened, days until sales close, and normalized sales-window progress.

Static show metadata was evaluated but excluded from the final feature set because it did not improve performance in preliminary experiments.

### Modelling and evaluation

XGBoost captures non-linear relationships in the engineered features. To predict an entire curve, the forecast for day *t + 1* is fed back into the feature engine before forecasting day *t + 2*.

Each ensemble member is trained on a bootstrap sample of complete shows. Predictions are aggregated into a mean forecast and a Student’s *t*-based 90% interval. The project reports MSE, MAE, Prediction Interval Coverage Probability (PICP), and Interval Score across several forecast horizons.

## Results

On held-out shows, the point forecasts generalized comparably to the training data:

| Forecast horizon | MAE | MSE | 90% interval coverage (PICP) |
| --- | ---: | ---: | ---: |
| 5 days | 0.004 | 4.5e-5 | 0.778 |
| 10 days | 0.007 | 1.0e-4 | 0.732 |
| 20 days | 0.013 | 4.0e-4 | 0.694 |
| 30 days | 0.019 | 0.001 | 0.672 |

MAE is measured on the cumulative sell-through fraction, so the 5-day and 30-day errors correspond to roughly 0.4 and 1.9 percentage points. Errors rise gradually with forecast horizon, as expected for an iterative method.

The current uncertainty intervals are under-calibrated: their empirical coverage is below the intended 90%. This is a documented limitation and a clear next step for the project—e.g., increasing ensemble diversity, adding data perturbation, or applying post-hoc interval calibration.

## Repository layout

```text
piccolo_teatro/
├── data_pipeline/       # Ingestion, cleaning, feature engineering, dataset creation
├── config/              # Feature switches and model configuration
├── train/               # Bayesian tuning, bootstrap training, evaluation metrics
├── trend_simulation/    # Iterative forecasting engine
├── models/              # Saved XGBoost parameters and ensembles
├── powerbi_visual/      # Power BI integration helpers
├── plotting.py          # Forecast visualisation utilities
└── get_metrics.py       # Metric aggregation
thesis/thesis.tex        # Full methodology and experimental analysis
```

## Running the project

### Prerequisites

- Python 3.11 or 3.12 (the project supports `>3.10, <3.13`)
- The three source CSV files supplied by the theatre: `D_SALES_LIST_SALES.csv`, `D_CONFIG_PROD_LIST.csv`, and `stagioni.csv`

The source data is proprietary and is therefore not included in this repository.

### Setup

```bash
git clone https://github.com/SiniscalchiCarlo/piccolo_teatro.git
cd piccolo_teatro
uv sync
cp .env.example .env
```

Set `FOLDER_PATH` in `.env` to the absolute directory containing the source CSV files and where generated datasets, forecasts, and metrics can be stored.

The training module also uses SciPy and scikit-optimize for interval computation and Bayesian hyperparameter search. Install them if they are not already present in your environment:

```bash
uv add scipy scikit-optimize
```

### Create datasets and train

```bash
# Clean the raw data, build show-level features, and create train/validation/test splits
uv run python -m piccolo_teatro.data_pipeline.create_model_dataset

# Tune (if needed), train/load the ensemble, and produce forecast outputs
uv run python -m piccolo_teatro.train.bootstrap_ensemble
```

Saved parameter files and ensembles are reused when available. Training from scratch may take time because it runs Bayesian search and fits 70 bootstrap models.

## Tech stack

Python · Pandas · NumPy · XGBoost · scikit-learn · scikit-optimize · SciPy · Matplotlib · Plotly · Power BI

## Documentation

The complete thesis—including the modelling rationale, feature definitions, experimental protocol, and full results—is available in [the LaTeX source](thesis/thesis.tex).

## Author

Carlo Siniscalchi · MSc thesis project, Mathematical Engineering, Politecnico di Milano (2024–2025)
