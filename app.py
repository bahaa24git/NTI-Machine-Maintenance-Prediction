from __future__ import annotations

import csv
import os
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from model_training import FAILURE_TARGET, ID_COLUMN, MODEL_NAMES, TARGET, get_feature_columns


DATA_PATH = Path(__file__).with_name("factory_sensor_simulator_2040.csv")
MODEL_PATH = Path(__file__).with_name("models") / "maintenance_models.joblib"
PREDICTION_LOG_PATH = Path(__file__).with_name("prediction_logs.csv")
LOG_COLUMNS = [
    "timestamp_utc",
    "model_version",
    "model",
    "source",
    "machine_id",
    "predicted_rul_days",
    "failure_probability_7d",
    "failure_risk",
    "urgency",
    "recommendation",
    "out_of_range_inputs",
]
RANDOM_STATE = 42
SNAPSHOT_YEAR = 2040
INTEGER_FEATURES = {
    "Installation_Year",
    "Operational_Hours",
    "Last_Maintenance_Days_Ago",
    "Maintenance_History_Count",
    "Failure_History_Count",
    "Error_Codes_Last_30_Days",
    "AI_Override_Events",
}

st.set_page_config(
    page_title="Machine Life Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(145deg, #07111f 0%, #0b1729 52%, #07111f 100%);}
    [data-testid="stSidebar"] {background: #081321; border-right: 1px solid #1d3348;}
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(18,40,59,.95), rgba(10,27,43,.95));
        border: 1px solid #24445e; border-radius: 14px; padding: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
    }
    .hero {padding: 1.25rem 0 .8rem 0;}
    .hero h1 {font-size: 2.35rem; margin: 0; letter-spacing: -.04em;}
    .hero p {color: #9db0c4; margin-top: .35rem; font-size: 1.02rem;}
    .status-chip {display:inline-block; padding:.28rem .7rem; border-radius:999px;
        background:#0d3b35; color:#70e1c1; border:1px solid #1f6358; font-size:.78rem;}
    .risk-low {color:#70e1c1}.risk-medium {color:#ffcd70}.risk-high {color:#ff7d8a}
    div[data-testid="stTabs"] button {font-weight: 650;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading fleet data…")
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_resource(show_spinner="Loading trained model artifacts…")
def load_artifact(path: str) -> dict:
    return joblib.load(path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return get_feature_columns(df)


def risk_label(rul: float) -> tuple[str, str]:
    if rul <= 30:
        return "Critical", "risk-high"
    if rul <= 90:
        return "Plan maintenance", "risk-medium"
    return "Healthy", "risk-low"


def format_days(value: float) -> str:
    return f"{max(value, 0):,.0f} days"


def filtered_fleet(df: pd.DataFrame, machine_types: list[str], max_rows: int = 12_000) -> pd.DataFrame:
    result = df[df["Machine_Type"].isin(machine_types)] if machine_types else df.iloc[0:0]
    if len(result) > max_rows:
        result = result.sample(max_rows, random_state=RANDOM_STATE)
    return result


def fleet_dashboard(df: pd.DataFrame, selected_types: list[str]) -> None:
    full_filtered = df[df["Machine_Type"].isin(selected_types)] if selected_types else df.iloc[0:0]
    if full_filtered.empty:
        st.warning("Select at least one machine type to populate the dashboard.")
        return

    critical = (full_filtered[TARGET] <= 30).sum()
    maintenance_due = (full_filtered[TARGET] <= 90).sum()
    cols = st.columns(5)
    cols[0].metric("Machines", f"{len(full_filtered):,}")
    cols[1].metric("Median remaining life", format_days(full_filtered[TARGET].median()))
    cols[2].metric("Critical ≤30 days", f"{critical:,}", f"{critical / len(full_filtered):.1%}")
    cols[3].metric("Due ≤90 days", f"{maintenance_due:,}", f"{maintenance_due / len(full_filtered):.1%}")
    cols[4].metric("Median operating hours", f"{full_filtered['Operational_Hours'].median():,.0f} h")

    sample = filtered_fleet(df, selected_types)
    left, right = st.columns((1.3, 1))
    with left:
        st.subheader("Remaining life vs. operating hours")
        scatter = px.scatter(
            sample,
            x="Operational_Hours",
            y=TARGET,
            color="Machine_Type",
            hover_data=[ID_COLUMN, "Temperature_C", "Vibration_mms"],
            opacity=0.55,
            labels={TARGET: "Remaining useful life (days)", "Operational_Hours": "Operating hours"},
            template="plotly_dark",
        )
        scatter.add_hline(y=30, line_dash="dot", line_color="#ff6174", annotation_text="Critical threshold")
        scatter.update_layout(showlegend=False, height=430, margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(scatter, width="stretch")

    with right:
        st.subheader("Machine-type comparison")
        by_type = (
            full_filtered.groupby("Machine_Type", as_index=False)
            .agg(
                Median_RUL=(TARGET, "median"),
                Median_Hours=("Operational_Hours", "median"),
                Machines=(ID_COLUMN, "count"),
            )
            .sort_values("Median_RUL")
        )
        comparison = px.bar(
            by_type,
            x="Median_RUL",
            y="Machine_Type",
            orientation="h",
            color="Median_RUL",
            color_continuous_scale=["#ff6174", "#ffcc66", "#45d4b4"],
            labels={"Median_RUL": "Median remaining life (days)", "Machine_Type": ""},
            template="plotly_dark",
        )
        comparison.update_layout(height=430, coloraxis_showscale=False, margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(comparison, width="stretch")

    st.subheader("Maintenance priority queue")
    priority = full_filtered.nsmallest(100, TARGET).copy()
    priority["Risk"] = priority[TARGET].map(lambda value: risk_label(value)[0])
    priority = priority[
        [ID_COLUMN, "Machine_Type", TARGET, "Risk", "Operational_Hours", "Last_Maintenance_Days_Ago", "Error_Codes_Last_30_Days"]
    ].rename(columns={TARGET: "Remaining life (days)"})
    st.dataframe(priority, width="stretch", hide_index=True, height=360)
    st.caption("Snapshot-based prioritization. The source has no timestamped sensor history, so this is not a degradation trajectory.")

    operational_insights(full_filtered)


def operational_insights(fleet: pd.DataFrame) -> None:
    st.subheader("Operational insights")
    st.caption("Associations within the selected fleet. These views describe the snapshot and do not establish cause and effect.")
    drivers_tab, operations_tab, lifecycle_tab = st.tabs(
        ["Risk drivers", "Maintenance & AI", "Lifecycle segments"]
    )

    with drivers_tab:
        sensor_columns = [
            "Operational_Hours",
            "Temperature_C",
            "Vibration_mms",
            "Sound_dB",
            "Oil_Level_pct",
            "Coolant_Level_pct",
            "Power_Consumption_kW",
            "Last_Maintenance_Days_Ago",
            "Failure_History_Count",
            "Error_Codes_Last_30_Days",
        ]
        correlations = (
            fleet[sensor_columns + [TARGET]]
            .corr(numeric_only=True)[TARGET]
            .drop(TARGET)
            .sort_values()
            .rename("Correlation")
            .rename_axis("Signal")
            .reset_index(name="Correlation")
        )
        left, right = st.columns(2)
        with left:
            correlation_chart = px.bar(
                correlations,
                x="Correlation",
                y="Signal",
                orientation="h",
                color="Correlation",
                color_continuous_scale=["#ff6174", "#203047", "#45d4b4"],
                range_color=[-1, 1],
                title="Correlation with remaining useful life",
                template="plotly_dark",
            )
            correlation_chart.update_layout(
                height=440, coloraxis_showscale=False, margin=dict(l=10, r=10, t=45, b=10)
            )
            st.plotly_chart(correlation_chart, width="stretch")

        with right:
            comparison_columns = [
                "Temperature_C",
                "Vibration_mms",
                "Sound_dB",
                "Oil_Level_pct",
                "Coolant_Level_pct",
                "Power_Consumption_kW",
                "Error_Codes_Last_30_Days",
            ]
            critical = fleet[fleet[TARGET] <= 30]
            healthy = fleet[fleet[TARGET] > 90]
            standard_deviation = fleet[comparison_columns].std().replace(0, np.nan)
            standardized_gap = (
                (critical[comparison_columns].mean() - healthy[comparison_columns].mean())
                / standard_deviation
            ).dropna().sort_values().rename("Standardized difference").rename_axis("Signal").reset_index(name="Standardized difference")
            gap_chart = px.bar(
                standardized_gap,
                x="Standardized difference",
                y="Signal",
                orientation="h",
                color="Standardized difference",
                color_continuous_scale=["#45d4b4", "#203047", "#ff6174"],
                range_color=[-0.15, 0.15],
                title="Critical vs. healthy sensor profile",
                template="plotly_dark",
            )
            gap_chart.update_layout(
                height=440, coloraxis_showscale=False, margin=dict(l=10, r=10, t=45, b=10)
            )
            st.plotly_chart(gap_chart, width="stretch")
        strongest = correlations.iloc[correlations["Correlation"].abs().argmax()]
        st.info(
            f"**Primary signal:** {strongest['Signal'].replace('_', ' ')} has a "
            f"{strongest['Correlation']:+.3f} correlation with remaining life. "
            "Small bars in the sensor-profile chart mean critical and healthy machines look similar on that signal."
        )

    with operations_tab:
        maintenance = fleet.copy()
        maintenance["Maintenance recency"] = pd.cut(
            maintenance["Last_Maintenance_Days_Ago"],
            bins=[-1, 30, 90, 180, np.inf],
            labels=["0–30 days", "31–90 days", "91–180 days", "181+ days"],
        )
        by_maintenance = (
            maintenance.groupby("Maintenance recency", observed=True)
            .agg(
                Machines=(ID_COLUMN, "size"),
                Median_RUL=(TARGET, "median"),
                Critical_rate=(TARGET, lambda values: 100 * (values <= 30).mean()),
            )
            .reset_index()
        )
        by_ai = (
            fleet.groupby("AI_Supervision", observed=True)
            .agg(
                Machines=(ID_COLUMN, "size"),
                Median_RUL=(TARGET, "median"),
                Critical_rate=(TARGET, lambda values: 100 * (values <= 30).mean()),
                Mean_errors=("Error_Codes_Last_30_Days", "mean"),
            )
            .reset_index()
        )
        by_ai["AI supervision"] = by_ai["AI_Supervision"].map({True: "Enabled", False: "Disabled"})
        left, right = st.columns(2)
        with left:
            maintenance_chart = px.bar(
                by_maintenance,
                x="Maintenance recency",
                y="Critical_rate",
                color="Median_RUL",
                color_continuous_scale=["#ff6174", "#45d4b4"],
                text_auto=".2f",
                title="Critical-risk rate by maintenance recency",
                labels={"Critical_rate": "Critical machines (%)"},
                template="plotly_dark",
            )
            maintenance_chart.update_layout(
                height=390, coloraxis_showscale=False, margin=dict(l=10, r=10, t=45, b=10)
            )
            st.plotly_chart(maintenance_chart, width="stretch")
        with right:
            ai_chart = px.bar(
                by_ai,
                x="AI supervision",
                y=["Critical_rate", "Mean_errors"],
                barmode="group",
                title="AI-supervised vs. unsupervised machines",
                labels={"value": "Rate / mean count", "variable": "Measure"},
                color_discrete_sequence=["#738cff", "#ffcc66"],
                template="plotly_dark",
            )
            ai_chart.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(ai_chart, width="stretch")
        ai_gap = by_ai.set_index("AI_Supervision")["Critical_rate"]
        if True in ai_gap and False in ai_gap:
            st.info(
                f"AI-supervised machines have a {ai_gap[True] - ai_gap[False]:+.2f} percentage-point "
                "difference in critical-risk rate. Treat this as an association; assignment to AI supervision may not be random."
            )

    with lifecycle_tab:
        lifecycle = fleet.copy()
        lifecycle["Machine age"] = (SNAPSHOT_YEAR - lifecycle["Installation_Year"]).clip(lower=0)
        lifecycle["Age group"] = pd.cut(
            lifecycle["Machine age"],
            bins=[-1, 5, 10, 20, np.inf],
            labels=["0–5 years", "6–10 years", "11–20 years", "21+ years"],
        )
        lifecycle["Utilization band"] = pd.qcut(
            lifecycle["Operational_Hours"], q=10, duplicates="drop"
        )
        by_utilization = (
            lifecycle.groupby("Utilization band", observed=True)
            .agg(Median_hours=("Operational_Hours", "median"), Median_RUL=(TARGET, "median"))
            .reset_index(drop=True)
        )
        by_age = (
            lifecycle.groupby("Age group", observed=True)
            .agg(
                Machines=(ID_COLUMN, "size"),
                Median_RUL=(TARGET, "median"),
                Median_hours=("Operational_Hours", "median"),
            )
            .reset_index()
        )
        left, right = st.columns(2)
        with left:
            utilization_chart = px.line(
                by_utilization,
                x="Median_hours",
                y="Median_RUL",
                markers=True,
                title="Remaining life across utilization bands",
                labels={"Median_hours": "Median operating hours", "Median_RUL": "Median RUL (days)"},
                template="plotly_dark",
            )
            utilization_chart.update_traces(line_color="#45d4b4", marker_size=9)
            utilization_chart.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(utilization_chart, width="stretch")
        with right:
            age_chart = px.bar(
                by_age,
                x="Age group",
                y="Median_RUL",
                color="Median_hours",
                color_continuous_scale=["#738cff", "#ffcc66"],
                title=f"Remaining life by machine age (snapshot year {SNAPSHOT_YEAR})",
                labels={"Median_RUL": "Median RUL (days)"},
                template="plotly_dark",
            )
            age_chart.update_layout(
                height=390, coloraxis_colorbar_title="Median hours", margin=dict(l=10, r=10, t=45, b=10)
            )
            st.plotly_chart(age_chart, width="stretch")
        st.caption(
            "Machine age uses 2040 because the source is named as a 2040 simulator and includes installation years through 2040."
        )


def maintenance_recommendation(rul: float, thresholds: dict[str, float]) -> tuple[str, str, int]:
    if rul <= thresholds["immediate"]:
        return "Stop and inspect immediately", "Critical", 0
    if rul <= thresholds["urgent"]:
        return "Schedule urgent maintenance", "High", min(7, max(1, int(rul / 2)))
    if rul <= thresholds["planned"]:
        return "Add to maintenance plan", "Medium", min(30, max(7, int(rul / 2)))
    return "Continue monitoring", "Low", min(60, max(30, int(rul / 3)))


def prediction_explanation(
    row: pd.DataFrame, pipeline, artifact: dict, predicted: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = artifact["reference"]
    effects = []
    unusual = []
    for column in artifact["features"]:
        counterfactual = row.copy()
        if column in reference["numeric"]:
            stats = reference["numeric"][column]
            value = float(row[column].iloc[0]) if pd.notna(row[column].iloc[0]) else stats["median"]
            counterfactual[column] = stats["median"]
            if value < stats["p05"] or value > stats["p95"]:
                unusual.append(
                    {
                        "Input": column.replace("_", " "),
                        "Value": value,
                        "Expected range": f"{stats['p05']:.2f}–{stats['p95']:.2f}",
                        "Flag": "Below normal" if value < stats["p05"] else "Above normal",
                    }
                )
        else:
            counterfactual[column] = reference["categorical"][column]
            value = row[column].iloc[0]
        counterfactual_prediction = max(float(pipeline.predict(counterfactual)[0]), 0)
        effects.append(
            {
                "Input": column.replace("_", " "),
                "Value": str(value),
                "Impact_days": predicted - counterfactual_prediction,
            }
        )
    influence = pd.DataFrame(effects)
    influence["Absolute impact"] = influence["Impact_days"].abs()
    influence = influence.nlargest(7, "Absolute impact").drop(columns="Absolute impact")
    return influence, pd.DataFrame(unusual)


def similar_machine_comparison(df: pd.DataFrame, row: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    machine_type = row["Machine_Type"].iloc[0]
    peers = df[df["Machine_Type"] == machine_type].copy()
    hours = float(row["Operational_Hours"].iloc[0])
    peers["_distance"] = (peers["Operational_Hours"] - hours).abs()
    nearest = peers.nsmallest(min(50, len(peers)), "_distance")
    metrics = [TARGET, "Operational_Hours", "Temperature_C", "Vibration_mms", "Maintenance_History_Count", "Error_Codes_Last_30_Days"]
    comparison = []
    for metric in metrics:
        value = float(row[metric].iloc[0]) if metric in row else np.nan
        distribution = peers[metric].dropna()
        percentile = 100 * (distribution <= value).mean() if len(distribution) else np.nan
        comparison.append(
            {
                "Metric": metric.replace("_", " "),
                "Machine": value,
                "Similar-machine median": float(nearest[metric].median()),
                "Type percentile": percentile,
            }
        )
    return pd.DataFrame(comparison), nearest


def read_prediction_logs(path: Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)
    rows: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        for values in reader:
            if len(values) == len(LOG_COLUMNS):
                rows.append(dict(zip(LOG_COLUMNS, values)))
            elif len(values) == len(header):
                rows.append(dict(zip(header, values)))
            elif values:
                skipped += 1
    logs = pd.DataFrame(rows).reindex(columns=LOG_COLUMNS)
    logs["failure_risk"] = logs["failure_risk"].fillna("Unknown")
    logs["out_of_range_inputs"] = pd.to_numeric(
        logs["out_of_range_inputs"], errors="coerce"
    ).fillna(0).astype(int)
    logs.attrs["skipped_rows"] = skipped
    return logs


def append_prediction_log(record: dict) -> None:
    existing = read_prediction_logs()
    updated = pd.concat(
        [existing, pd.DataFrame([{column: record.get(column) for column in LOG_COLUMNS}])],
        ignore_index=True,
    )
    temporary_path = PREDICTION_LOG_PATH.with_suffix(".tmp")
    updated.to_csv(temporary_path, columns=LOG_COLUMNS, index=False)
    temporary_path.replace(PREDICTION_LOG_PATH)


def render_prediction_result(
    df: pd.DataFrame,
    row: pd.DataFrame,
    artifact: dict,
    model_name: str,
    thresholds: dict[str, float],
    source: str,
    actual: float | None = None,
) -> None:
    bundle = artifact["regression"][model_name]
    pipeline = bundle["pipeline"]
    features = artifact["features"]
    predicted = max(float(pipeline.predict(row[features])[0]), 0)
    failure_probability = float(
        artifact["failure_classifier"]["pipeline"].predict_proba(row[features])[0, 1]
    )
    recommendation, urgency, inspection_days = maintenance_recommendation(predicted, thresholds)
    if failure_probability >= thresholds["failure_probability"] and urgency in {"Low", "Medium"}:
        recommendation = "Schedule urgent inspection based on elevated 7-day failure risk"
        urgency = "High"
        inspection_days = min(inspection_days, 2)
    if failure_probability >= 0.70:
        failure_risk = "Critical"
    elif failure_probability >= thresholds["failure_probability"]:
        failure_risk = "High"
    elif failure_probability >= 0.15:
        failure_risk = "Moderate"
    else:
        failure_risk = "Low"
    inspection_date = date.today() + timedelta(days=inspection_days)
    rmse = bundle["metrics"]["RMSE"]
    lower = max(0, predicted - 1.96 * rmse)
    upper = predicted + 1.96 * rmse

    cards = st.columns(3)
    cards[0].metric("Predicted remaining life", format_days(predicted))
    cards[1].metric("7-day failure probability", f"{failure_probability:.1%}")
    cards[2].metric("Failure risk", failure_risk)
    cards = st.columns(3)
    cards[0].metric("Maintenance urgency", urgency)
    cards[1].metric("Suggested inspection", inspection_date.strftime("%d %b %Y"))
    cards[2].metric("Expected model error", f"±{bundle['metrics']['MAE']:.0f} days")
    st.warning(f"**Recommended action:** {recommendation}") if urgency in {"Critical", "High"} else st.info(f"**Recommended action:** {recommendation}")
    st.caption(
        f"Approximate 95% error band: {lower:.0f}–{upper:.0f} days, based on validation RMSE. "
        "This is an empirical error range, not a calibrated probability interval."
    )
    if actual is not None:
        st.metric("Dataset RUL", format_days(actual), f"{predicted - actual:+.0f} day prediction gap")

    influence, unusual = prediction_explanation(row[features], pipeline, artifact, predicted)
    explain_tab, unusual_tab, peers_tab = st.tabs(
        ["Why this prediction", "Unusual inputs", "Similar machines"]
    )
    with explain_tab:
        chart = px.bar(
            influence.sort_values("Impact_days"),
            x="Impact_days",
            y="Input",
            orientation="h",
            color="Impact_days",
            color_continuous_scale=["#ff6174", "#203047", "#45d4b4"],
            labels={"Impact_days": "Estimated local impact (days)"},
            template="plotly_dark",
        )
        chart.update_layout(height=390, coloraxis_showscale=False, margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(chart, width="stretch")
        st.caption("Model-agnostic local influence: each input is replaced with the fleet median/mode while other inputs stay fixed. It is not a causal effect.")
    with unusual_tab:
        if unusual.empty:
            st.success("No numeric inputs fall outside the training data’s 5th–95th percentile range.")
        else:
            st.dataframe(unusual, width="stretch", hide_index=True)
    with peers_tab:
        if source == "dataset":
            comparison, peers = similar_machine_comparison(df, row)
        else:
            enriched = row.copy()
            enriched[TARGET] = predicted
            comparison, peers = similar_machine_comparison(df, enriched)
        st.dataframe(
            comparison.style.format({"Machine": "{:.2f}", "Similar-machine median": "{:.2f}", "Type percentile": "{:.1f}%"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Compared with the 50 closest {row['Machine_Type'].iloc[0]} machines by operational hours.")

    machine_id = str(row[ID_COLUMN].iloc[0]) if ID_COLUMN in row else "CUSTOM"
    log_record = {
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_version": artifact["version"],
        "model": model_name,
        "source": source,
        "machine_id": machine_id,
        "predicted_rul_days": round(predicted, 2),
        "failure_probability_7d": round(failure_probability, 6),
        "failure_risk": failure_risk,
        "urgency": urgency,
        "recommendation": recommendation,
        "out_of_range_inputs": int(len(unusual)),
    }
    append_prediction_log(log_record)
    report = pd.DataFrame([log_record | {"inspection_date": inspection_date.isoformat(), "interval_low": round(lower, 1), "interval_high": round(upper, 1)}])
    st.download_button(
        "Download prediction report",
        report.to_csv(index=False).encode("utf-8"),
        file_name=f"prediction_{machine_id}.csv",
        mime="text/csv",
    )


def model_comparison(artifact: dict) -> None:
    regression = artifact["regression"]
    metrics = pd.DataFrame(
        [
            {
                "Model": name,
                **bundle["metrics"],
                "Train R²": bundle["train_metrics"]["R²"],
                "Critical MAE": bundle["critical_mae"],
                "CV MAE": bundle["cv_mae_mean"],
                "CV variation": bundle["cv_mae_std"],
                "Overfit warning": bundle["overfit_warning"],
            }
            for name, bundle in regression.items()
        ]
    ).sort_values("MAE")
    best = metrics.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Best validation model", best["Model"])
    cols[1].metric("Best MAE", format_days(best["MAE"]))
    cols[2].metric("Best R²", f"{best['R²']:.3f}")
    cols[3].metric("7-day classifier AUC", f"{artifact['failure_classifier']['metrics']['ROC AUC']:.3f}")

    chart = px.bar(
        metrics,
        x="Model",
        y=["MAE", "RMSE"],
        barmode="group",
        labels={"value": "Error (days)", "variable": "Metric"},
        color_discrete_sequence=["#45d4b4", "#738cff"],
        template="plotly_dark",
    )
    chart.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(chart, width="stretch")
    st.dataframe(metrics.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}", "Train R²": "{:.4f}", "Critical MAE": "{:.2f}", "CV MAE": "{:.2f}", "CV variation": "{:.2f}"}), width="stretch", hide_index=True)

    selected = st.selectbox("Detailed evaluation model", list(regression), key="evaluation_model")
    bundle = regression[selected]
    predictions = bundle["predictions"]
    actual_tab, residual_tab, segment_tab, robustness_tab, neural_tab = st.tabs(
        ["Actual vs predicted", "Error distribution", "Machine types", "Split robustness", "Learning curve"]
    )
    with actual_tab:
        figure = px.scatter(predictions, x="Actual", y="Predicted", color="Machine_Type", opacity=0.5, template="plotly_dark")
        bound = float(max(predictions["Actual"].max(), predictions["Predicted"].max()))
        figure.add_trace(go.Scatter(x=[0, bound], y=[0, bound], mode="lines", name="Perfect prediction", line=dict(dash="dash", color="#ffffff")))
        figure.update_layout(height=430, showlegend=False)
        st.plotly_chart(figure, width="stretch")
    with residual_tab:
        figure = px.histogram(predictions, x="Residual", nbins=60, template="plotly_dark", labels={"Residual": "Actual − predicted (days)"})
        figure.add_vline(x=0, line_dash="dash", line_color="#ffffff")
        st.plotly_chart(figure, width="stretch")
    with segment_tab:
        by_type = bundle["by_machine_type"].sort_values("MAE", ascending=False)
        st.plotly_chart(px.bar(by_type, x="MAE", y="Machine_Type", orientation="h", color="MAE", template="plotly_dark"), width="stretch")
        st.dataframe(by_type, width="stretch", hide_index=True)
    with robustness_tab:
        robustness = pd.DataFrame([{"Split": split, **values} for split, values in bundle["robustness"].items()])
        st.dataframe(robustness, width="stretch", hide_index=True)
        st.caption(artifact["notes"]["time_split"])
    with neural_tab:
        history = bundle["learning_history"]
        if history:
            learning = pd.DataFrame({key: pd.Series(value) for key, value in history.items()})
            learning.index += 1
            learning.index.name = "Iteration"
            st.line_chart(learning, width="stretch")
        else:
            st.info("Learning history is available when Neural Network is selected.")

    classifier = artifact["failure_classifier"]["metrics"]
    st.subheader("7-day failure classifier")
    st.dataframe(pd.DataFrame([classifier]).style.format("{:.3f}"), width="stretch", hide_index=True)
    st.caption(f"Model version {artifact['version']} · trained {artifact['trained_at']} · {artifact['training_rows']:,} training records")


def dataset_prediction(df: pd.DataFrame, artifact: dict, model_name: str, thresholds: dict[str, float]) -> None:
    left, right = st.columns((1, 2))
    with left:
        machine_id = st.text_input("Machine ID", value=str(df[ID_COLUMN].iloc[0]), help="Example: MC_000000")
        run = st.button("Predict selected machine", type="primary", width="stretch")
    matches = df[df[ID_COLUMN].astype(str).str.upper() == machine_id.strip().upper()]
    with right:
        if run and matches.empty:
            st.error("Machine ID was not found in the dataset.")
        elif run:
            row = matches.iloc[[0]]
            actual = float(row[TARGET].iloc[0])
            render_prediction_result(df, row, artifact, model_name, thresholds, "dataset", actual)


def custom_prediction(df: pd.DataFrame, artifact: dict, model_name: str, thresholds: dict[str, float]) -> None:
    features = artifact["features"]
    categorical = df[features].select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    numeric = [column for column in features if column not in categorical]

    with st.form("custom_machine_form"):
        st.caption("Enter the current sensor snapshot. Optional equipment-specific readings default to the fleet median.")
        values: dict[str, object] = {}
        columns = st.columns(3)
        for index, column in enumerate(features):
            with columns[index % 3]:
                if column in categorical:
                    options = sorted(df[column].dropna().unique().tolist(), key=str)
                    values[column] = st.selectbox(column.replace("_", " "), options, key=f"custom_{column}")
                else:
                    series = df[column].dropna()
                    median = float(series.median()) if not series.empty else 0.0
                    if column in INTEGER_FEATURES:
                        values[column] = st.number_input(
                            column.replace("_", " "),
                            value=int(round(median)),
                            step=1,
                            format="%d",
                            key=f"custom_{column}",
                        )
                    else:
                        values[column] = st.number_input(
                            column.replace("_", " "),
                            value=median,
                            key=f"custom_{column}",
                        )
        submitted = st.form_submit_button("Estimate remaining useful life", type="primary", width="stretch")

    if submitted:
        input_frame = pd.DataFrame([values], columns=features)
        render_prediction_result(df, input_frame, artifact, model_name, thresholds, "custom")


def batch_prediction(df: pd.DataFrame, artifact: dict, model_name: str, thresholds: dict[str, float]) -> None:
    st.write("Upload a CSV containing the model feature columns. Extra columns are preserved in the download.")
    uploaded = st.file_uploader("Machine sensor CSV", type="csv")
    if uploaded is None:
        with st.expander("Required columns"):
            st.code("\n".join(feature_columns(df)))
        return
    try:
        batch = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"The CSV could not be read: {exc}")
        return
    features = artifact["features"]
    missing = [column for column in features if column not in batch.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return
    scored = batch.copy()
    scored["Predicted_Remaining_Useful_Life_days"] = np.maximum(
        artifact["regression"][model_name]["pipeline"].predict(batch[features]), 0
    ).round(1)
    scored["Failure_Probability_Within_7_Days"] = artifact["failure_classifier"]["pipeline"].predict_proba(batch[features])[:, 1].round(4)
    scored["Maintenance_Recommendation"] = scored["Predicted_Remaining_Useful_Life_days"].map(lambda value: maintenance_recommendation(value, thresholds)[0])
    numeric_reference = artifact["reference"]["numeric"]
    out_of_range = pd.DataFrame(index=batch.index)
    for column, stats in numeric_reference.items():
        values = pd.to_numeric(batch[column], errors="coerce")
        out_of_range[column] = (values < stats["p05"]) | (values > stats["p95"])
    scored["Out_of_range_feature_count"] = out_of_range.sum(axis=1)
    drift_rate = float(out_of_range.to_numpy().mean()) if not out_of_range.empty else 0.0
    st.success(f"Scored {len(scored):,} machines.")
    st.metric("Values outside training 5th–95th range", f"{drift_rate:.1%}")
    if drift_rate > 0.20:
        st.warning("This batch differs materially from the training reference. Review model performance before operational use.")
    st.dataframe(scored.head(200), width="stretch", hide_index=True)
    st.download_button(
        "Download predictions",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="machine_life_predictions.csv",
        mime="text/csv",
        width="stretch",
    )


def prediction_workspace(df: pd.DataFrame, artifact: dict, model_name: str, thresholds: dict[str, float]) -> None:
    bundle = artifact["regression"][model_name]
    st.info(f"Active model: **{model_name}** · validation MAE **{bundle['metrics']['MAE']:.1f} days** · R² **{bundle['metrics']['R²']:.3f}**")
    dataset_tab, custom_tab, batch_tab = st.tabs(["Select from dataset", "Enter custom data", "Batch CSV"])
    with dataset_tab:
        dataset_prediction(df, artifact, model_name, thresholds)
    with custom_tab:
        custom_prediction(df, artifact, model_name, thresholds)
    with batch_tab:
        batch_prediction(df, artifact, model_name, thresholds)


def machine_comparison(df: pd.DataFrame, artifact: dict, model_name: str) -> None:
    st.subheader("Compare machines")
    default_ids = ", ".join(df[ID_COLUMN].astype(str).head(3))
    requested = st.text_input(
        "Machine IDs (comma-separated, maximum 8)",
        value=default_ids,
        help="Example: MC_000000, MC_000001",
    )
    ids = list(dict.fromkeys(value.strip() for value in requested.split(",") if value.strip()))[:8]
    selected = df[df[ID_COLUMN].astype(str).isin(ids)].copy()
    missing = [machine_id for machine_id in ids if machine_id not in set(selected[ID_COLUMN].astype(str))]
    if missing:
        st.warning("Not found: " + ", ".join(missing))
    if selected.empty:
        st.info("Enter at least one valid machine ID.")
        return

    features = artifact["features"]
    selected["Predicted_RUL"] = np.maximum(
        artifact["regression"][model_name]["pipeline"].predict(selected[features]), 0
    )
    selected["Failure_probability_7d"] = artifact["failure_classifier"]["pipeline"].predict_proba(selected[features])[:, 1]
    table_columns = [
        ID_COLUMN, "Machine_Type", TARGET, "Predicted_RUL", "Failure_probability_7d",
        "Operational_Hours", "Temperature_C", "Vibration_mms",
        "Maintenance_History_Count", "Error_Codes_Last_30_Days",
    ]
    st.dataframe(
        selected[table_columns].style.format(
            {TARGET: "{:.0f}", "Predicted_RUL": "{:.1f}", "Failure_probability_7d": "{:.1%}"}
        ),
        width="stretch",
        hide_index=True,
    )

    metrics = [
        "Operational_Hours", "Temperature_C", "Vibration_mms",
        "Maintenance_History_Count", "Error_Codes_Last_30_Days", TARGET,
    ]
    radar = go.Figure()
    labels = [metric.replace("_", " ") for metric in metrics]
    for _, machine in selected.iterrows():
        percentiles = []
        peers = df[df["Machine_Type"] == machine["Machine_Type"]]
        for metric in metrics:
            distribution = peers[metric].dropna()
            percentiles.append(float(100 * (distribution <= machine[metric]).mean()))
        radar.add_trace(
            go.Scatterpolar(
                r=percentiles + [percentiles[0]],
                theta=labels + [labels[0]],
                fill="toself",
                name=str(machine[ID_COLUMN]),
                opacity=0.65,
            )
        )
    radar.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
        title="Percentile within each machine type",
        height=520,
    )
    st.plotly_chart(radar, width="stretch")
    st.caption("Each axis is a percentile relative to machines of the same type, making different units directly comparable.")


def model_monitor(artifact: dict) -> None:
    st.subheader("Model operations monitor")
    st.caption(f"Active artifact: {artifact['version']} · trained {artifact['trained_at']}")
    if not PREDICTION_LOG_PATH.exists():
        st.info("No prediction events have been logged yet. Run a dataset or custom prediction first.")
        return
    try:
        logs = read_prediction_logs()
    except (OSError, csv.Error) as exc:
        st.error(f"Prediction log could not be read: {exc}")
        return
    if logs.empty:
        st.info("No valid prediction events were found in the log.")
        return
    skipped_rows = logs.attrs.get("skipped_rows", 0)
    if skipped_rows:
        st.warning(f"Skipped {skipped_rows} malformed prediction-log row(s).")
    logs["timestamp_utc"] = pd.to_datetime(logs["timestamp_utc"], utc=True, errors="coerce")
    logs["failure_probability_7d"] = pd.to_numeric(logs["failure_probability_7d"], errors="coerce")
    logs["date"] = logs["timestamp_utc"].dt.date
    cols = st.columns(4)
    cols[0].metric("Logged predictions", f"{len(logs):,}")
    cols[1].metric("Mean 7-day risk", f"{logs['failure_probability_7d'].mean():.1%}")
    cols[2].metric("High/critical urgency", f"{logs['urgency'].isin(['High', 'Critical']).mean():.1%}")
    outlier_rate = (logs.get("out_of_range_inputs", pd.Series(0, index=logs.index)) > 0).mean()
    cols[3].metric("Predictions with unusual inputs", f"{outlier_rate:.1%}")
    left, right = st.columns(2)
    with left:
        daily = logs.groupby("date", as_index=False).agg(Predictions=("model", "size"), Mean_failure_risk=("failure_probability_7d", "mean"))
        st.plotly_chart(px.bar(daily, x="date", y="Predictions", title="Prediction volume", template="plotly_dark"), width="stretch")
    with right:
        urgency = logs["urgency"].value_counts().rename_axis("Urgency").reset_index(name="Predictions")
        st.plotly_chart(px.bar(urgency, x="Urgency", y="Predictions", color="Urgency", title="Maintenance urgency mix", template="plotly_dark"), width="stretch")
    st.dataframe(logs.sort_values("timestamp_utc", ascending=False).head(200), width="stretch", hide_index=True)
    st.download_button("Download prediction log", logs.to_csv(index=False).encode("utf-8"), file_name="prediction_logs.csv", mime="text/csv")
    st.caption("Out-of-range monitoring is an early drift indicator. Formal drift monitoring requires enough production predictions and later outcome labels.")


def main() -> None:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH.name}. Place it beside app.py and restart the app.")
        st.stop()
    try:
        df = load_data(str(DATA_PATH))
    except Exception as exc:
        st.error(f"Could not load the dataset: {exc}")
        st.stop()
    required = {ID_COLUMN, TARGET, "Machine_Type", "Operational_Hours"}
    missing = required.difference(df.columns)
    if missing:
        st.error("Dataset is missing required columns: " + ", ".join(sorted(missing)))
        st.stop()

    if not MODEL_PATH.exists():
        st.error(
            "Trained models were not found. Run `python train_models.py` once, then restart Streamlit."
        )
        st.stop()
    try:
        artifact = load_artifact(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Could not load trained models: {exc}")
        st.stop()
    with st.sidebar:
        st.markdown("## ⚙️ Machine Life")
        page = st.radio("Workspace", ["Fleet dashboard", "Predict lifespan", "Machine comparison", "Compare models", "Model monitor"])
        model_name = st.selectbox("Prediction model", MODEL_NAMES, index=0)
        with st.expander("Maintenance thresholds"):
            immediate = st.number_input("Immediate action ≤ days", min_value=1, max_value=30, value=7, step=1)
            urgent = st.number_input("Urgent maintenance ≤ days", min_value=immediate + 1, max_value=90, value=max(30, immediate + 1), step=1)
            planned = st.number_input("Maintenance plan ≤ days", min_value=urgent + 1, max_value=365, value=max(90, urgent + 1), step=1)
            failure_alert_pct = st.number_input("Failure alert probability (%)", min_value=10, max_value=90, value=50, step=5)
        thresholds = {
            "immediate": int(immediate),
            "urgent": int(urgent),
            "planned": int(planned),
            "failure_probability": float(failure_alert_pct) / 100,
        }
        st.divider()
        machine_types = sorted(df["Machine_Type"].dropna().unique().tolist())
        selected_types = st.multiselect("Machine types", machine_types, default=machine_types)
        st.divider()
        st.caption(
            f"Source: {DATA_PATH.name}\n\n{len(df):,} machine snapshots · {len(machine_types)} types"
            f"\n\nModel: {artifact['version']}"
        )

    st.markdown(
        """<div class='hero'><span class='status-chip'>PREDICTIVE MAINTENANCE</span>
        <h1>Machine Life Intelligence</h1>
        <p>Monitor fleet health, compare equipment, and estimate remaining useful life from live-style sensor inputs.</p></div>""",
        unsafe_allow_html=True,
    )

    if page == "Fleet dashboard":
        fleet_dashboard(df, selected_types)
    elif page == "Predict lifespan":
        prediction_workspace(df, artifact, model_name, thresholds)
    elif page == "Machine comparison":
        machine_comparison(df, artifact, model_name)
    elif page == "Compare models":
        model_comparison(artifact)
    else:
        model_monitor(artifact)


if __name__ == "__main__":
    main()
