# Machine Maintenance Prediction Dashboard

An interactive Streamlit app for exploring fleet health, comparing regression models, and predicting remaining useful life (RUL) from dataset rows, custom sensor readings, or uploaded CSV files.

## Run locally

```powershell
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

The file `factory_sensor_simulator_2040.csv` must be in the same directory as `app.py`.

## App workspaces

- **Fleet dashboard:** fleet KPIs, operating-hours vs. RUL chart, machine-type comparison, maintenance priority queue, risk-driver analysis, maintenance and AI associations, and lifecycle segments.
- **Predict lifespan:** RUL, 7-day failure probability, uncertainty, local input influence, unusual inputs, peer comparison, maintenance action, logging, and a downloadable report.
- **Machine comparison:** compare up to eight machine IDs using raw measures and within-type percentiles.
- **Compare models:** training/test metrics, actual-vs-predicted, residuals, cross-validation, critical-machine error, machine-type performance, holdout robustness, overfitting flags, and neural learning history.
- **Model monitor:** prediction volume, risk and urgency mix, unusual-input monitoring, recent events, and log download.

## Modeling notes

- The target is `Remaining_Useful_Life_days`.
- `Machine_ID` is an identifier and is not used as a feature.
- `Failure_Within_7_Days` is excluded because it is derived from the target and would leak future information.
- `train_models.py` trains on a reproducible sample of up to 60,000 records and saves `models/maintenance_models.joblib`; Streamlit only loads the artifact.
- Numeric inputs are standardized for stable neural-network training.
- The neural network also scales its target and uses early stopping.
- A separate balanced Random Forest classifier predicts `Failure_Within_7_Days`; the RUL target is excluded from its inputs.
- Prediction uncertainty is an empirical range derived from validation RMSE, not a calibrated probability interval.
- The dataset is a cross-sectional snapshot. It supports RUL comparison, but not a true sensor degradation timeline.

## Docker

Train the artifact before building:

```powershell
python train_models.py
docker build -t machine-life-dashboard .
docker run --rm -p 8501:8501 machine-life-dashboard
```

## Production extension points

- Predictions append to `prediction_logs.csv`; replace this with a database repository for multi-user deployment.
- Add authentication at the reverse proxy or hosting platform.
- Email or Slack alerts require credentials and an approved delivery destination.
- True degradation curves and time-based validation require timestamped sensor history, which this snapshot dataset does not contain.
