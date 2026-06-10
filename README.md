# entsoe_predict

Forecasting pipeline for German/Luxembourg day-ahead electricity prices using
ENTSO-E data, engineered time-series features, an XGBoost baseline model, MLflow
model storage, and a FastAPI inference endpoint.

## What This Project Does

The project fetches ENTSO-E market and grid data for the `DE_LU` bidding zone,
builds a feature table, trains a baseline model, and exposes a forecast API.

The tracked code covers:

- fetching day-ahead prices, actual renewable generation, renewable forecasts,
  and load forecasts from ENTSO-E
- merging fetched data into a single time-indexed dataset
- building cyclical time features, lagged prices, and renewable interaction
  features
- training an `XGBRFRegressor` baseline and logging metrics/model artifacts to
  MLflow
- loading an MLflow pyfunc model at API startup and returning predictions from a
  `/forecast` endpoint

## Repository Layout

| File | Purpose |
| --- | --- |
| `api_request.py` | FastAPI app that loads a model from `S3_URL`, fetches fresh data, builds inference features, and serves `/forecast`. |
| `fetch_data.py` | ENTSO-E data collection and merge utilities. |
| `features.py` | Feature engineering and inference feature validation. |
| `config.py` | Expected feature column order used before inference. |
| `baseline_model.py` | Training script for the XGBoost random forest baseline with MLflow logging. |
| `Dockerfile` | Container definition for running the FastAPI service with Uvicorn. |
| `LICENSE` | MIT license. |

Generated folders such as `logs/`, `fetched_data/`, and `gold/` are created by
the scripts at runtime.

## Data Pipeline

1. `fetch_data.py` queries ENTSO-E using `EntsoePandasClient`.
2. The fetch step collects:
   - day-ahead prices
   - forecasted load
   - actual solar, offshore wind, and onshore wind generation
   - forecasted solar, offshore wind, and onshore wind generation
3. `data_merger()` checks that all inputs share the same timestamp index and
   saves the merged dataset under `fetched_data/merged/`.
4. `feature_builder()` converts timestamps to `Europe/Brussels`, creates
   cyclical calendar features, adds a 24-step lagged price feature, creates solar
   interaction features, and saves the final feature table under `gold/`.
5. `prepare_inference_features()` validates that columns match
   `EXPECTED_COLUMNS_AND_ORDER`, then drops `day_ahead_price` before prediction.

## Environment Variables

The code reads environment variables with `python-dotenv`.

| Variable | Used By | Description |
| --- | --- | --- |
| `ENTENSO_API` | `fetch_data.py` | ENTSO-E API key used by `EntsoePandasClient`. |
| `S3_URL` | `api_request.py` | MLflow model URI loaded with `mlflow.pyfunc.load_model()`. |

## Running the API

The FastAPI app is defined in `api_request.py`.

```bash
uvicorn api_request:app --host 0.0.0.0 --port 8000
```

At startup, the app loads the model from `S3_URL`. Forecast requests are sent to
`POST /forecast`.

Example request body:

```json
{
  "start_time": "2026-01-01",
  "horizon": 1
}
```

Example response shape:

```json
{
  "start_date": "2026-01-01",
  "horizon": 1,
  "predictions": [0.0],
  "timestamps": ["2026-01-01 00:00:00+01:00"]
}
```

## Training the Baseline Model

`baseline_model.py` trains an `xgboost.XGBRFRegressor` on a feature CSV, reports
MAE, RMSE, and R2, logs metrics to MLflow, and registers the trained model.
MAE 22.29 EUR/MWh vs naive baseline of 43.59.

```bash
python baseline_model.py
```

The script currently expects a feature dataset at:

```text
gold/features_2026-01-01_to_2026-05-31
```

## Docker

The tracked `Dockerfile` builds a Python 3.11 image, installs dependencies from
`requirements.txt`, copies the API and data-processing modules, exposes port
`8000`, and starts Uvicorn. The build expects `requirements.txt` to be present
in the Docker build context.

```bash
docker build -t entsoe-predict .
docker run -p 8000:8000 --env ENTENSO_API=... --env S3_URL=... entsoe-predict
```

## License

This project is licensed under the MIT License.
