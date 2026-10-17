"""Read-only visualization of frozen forecasting evaluation artifacts."""

import plotly.express as px
import streamlit as st


def prepare_forecast_series(predictions, model, store_id, product_id):
    """Select saved observations for one model and store-product series."""
    return predictions.loc[
        predictions["model"].eq(model)
        & predictions["store_id"].eq(store_id)
        & predictions["product_id"].eq(product_id)
    ].sort_values("date").copy()


def render_forecast_performance(data):
    """Render saved test results and validation explainability."""
    metrics = data["holdout_metrics"].copy()
    predictions = data["holdout_predictions"].copy()
    importance = data["permutation_importance"].copy()
    summary = data["holdout_summary"]
    selection = data["champion_selection"]

    champion = selection["champion_model"]
    baseline = "rolling_mean_7"

    champion_row = metrics.loc[metrics["model"].eq(champion)].iloc[0]
    baseline_row = metrics.loc[metrics["model"].eq(baseline)].iloc[0]

    relative_improvement = (
        (baseline_row["wape_pct"] - champion_row["wape_pct"])
        / baseline_row["wape_pct"]
        * 100
    )

    st.title("Forecast Performance")
    st.caption(
        "Frozen model evaluation | Historical synthetic data | "
        "Saved artifacts only"
    )

    st.info(
        "The champion was selected using validation data before the "
        "held-out test was evaluated once. This page only visualizes "
        "saved results. The protocol is rolling one-day-ahead, using "
        "actual recent observations for lagged features; it is not "
        "a fixed-origin multi-day forecast."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Champion Test WAPE", f"{champion_row['wape_pct']:.4f}%")
    col2.metric("Baseline Test WAPE", f"{baseline_row['wape_pct']:.4f}%")
    col3.metric("Relative WAPE Improvement", f"{relative_improvement:.2f}%")
    col4.metric("Champion Test RMSSE", f"{champion_row['rmsse']:.4f}")

    st.caption(
        f"Champion: {champion} | "
        f"Baseline: {baseline} | "
        f"Test observations per model: {int(champion_row['observations']):,} | "
        f"Forecast horizon: {summary['forecast_horizon_days']} day"
    )

    st.subheader("Saved Holdout Model Comparison")

    comparison = metrics.melt(
        id_vars="model",
        value_vars=["wape_pct", "rmsse", "bias_pct"],
        var_name="metric",
        value_name="value",
    )

    metric_labels = {
        "wape_pct": "WAPE (%)",
        "rmsse": "RMSSE",
        "bias_pct": "Bias (%)",
    }
    selected_metric = st.selectbox(
        "Comparison metric",
        options=list(metric_labels),
        format_func=metric_labels.get,
    )

    metric_chart = px.bar(
        comparison.loc[comparison["metric"].eq(selected_metric)],
        x="model",
        y="value",
        color="model",
        labels={
            "model": "Saved Model",
            "value": metric_labels[selected_metric],
        },
        title=f"Holdout {metric_labels[selected_metric]}",
    )
    metric_chart.update_layout(
        height=380,
        showlegend=False,
    )
    st.plotly_chart(metric_chart, width="stretch")

    st.dataframe(
        metrics,
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Actual vs Saved Predictions")

    store_ids = sorted(predictions["store_id"].unique().tolist())
    product_ids = sorted(predictions["product_id"].unique().tolist())

    first, second, third = st.columns(3)

    with first:
        selected_store = st.selectbox(
            "Store",
            options=store_ids,
            key="forecast_store",
        )
    with second:
        selected_product = st.selectbox(
            "Product",
            options=product_ids,
            key="forecast_product",
        )
    with third:
        selected_model = st.selectbox(
            "Saved prediction model",
            options=[champion, baseline],
            key="forecast_model",
        )

    series = prepare_forecast_series(
        predictions,
        selected_model,
        selected_store,
        selected_product,
    )

    if series.empty:
        st.warning("No saved predictions for this selection.")
    else:
        chart_data = series.melt(
            id_vars="date",
            value_vars=["units_sold", "prediction"],
            var_name="series",
            value_name="units",
        )
        chart_data["series"] = chart_data["series"].replace(
            {
                "units_sold": "Actual Units Sold",
                "prediction": "Saved Prediction",
            }
        )

        forecast_chart = px.line(
            chart_data,
            x="date",
            y="units",
            color="series",
            labels={
                "date": "Date",
                "units": "Units",
                "series": "Series",
            },
            title=(
                f"{selected_store} / {selected_product} — "
                f"{selected_model}"
            ),
        )
        forecast_chart.update_layout(
            height=440,
            hovermode="x unified",
        )
        st.plotly_chart(forecast_chart, width="stretch")

        st.caption(
            "Historical held-out observations and previously saved "
            "one-day-ahead predictions. Changing filters only selects "
            "existing rows; it does not run model inference."
        )

        st.dataframe(
            series[
                ["date", "units_sold", "prediction", "error"]
            ],
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.subheader("Validation Feature Importance")

    importance = importance.sort_values(
        "wape_increase_mean_pp",
        ascending=True,
    )

    importance_chart = px.bar(
        importance,
        x="wape_increase_mean_pp",
        y="feature",
        orientation="h",
        labels={
            "wape_increase_mean_pp": (
                "Mean Validation WAPE Increase (percentage points)"
            ),
            "feature": "Feature",
        },
        title="Permutation Importance on Validation Data",
    )
    importance_chart.update_layout(height=520)
    st.plotly_chart(importance_chart, width="stretch")

    st.caption(
        "Permutation importance was measured on the validation split "
        "using the frozen model. Larger positive WAPE increases "
        "indicate greater reliance on the feature under this test. "
        "These values are not causal effects."
    )

    with st.expander("Evaluation provenance and limitations"):
        st.write(
            {
                "champion": champion,
                "selection_metric": selection["selection_metric"],
                "selection_status": selection["selection_status"],
                "evaluation_protocol": summary["evaluation_protocol"],
                "forecast_horizon_days": summary[
                    "forecast_horizon_days"
                ],
                "rmsse_scale_source": summary["rmsse_scale_source"],
                "test_used_for_model_selection": summary[
                    "test_used_for_model_selection"
                ],
                "retraining_performed": summary[
                    "retraining_performed"
                ],
                "data_source": summary["data_source"],
            }
        )
        st.warning(
            "The held-out test results are reported descriptively, "
            "not used to revise the frozen champion. Results come "
            "from synthetic retail data and do not establish "
            "production forecasting performance."
        )