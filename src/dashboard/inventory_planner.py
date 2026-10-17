"""Read-only visualization of saved inventory planning recommendations."""

import plotly.express as px
import streamlit as st

STATUS_ORDER = [
    "STOCKOUT",
    "BELOW_LEAD_TIME_COVERAGE",
    "REPLENISHMENT_SUGGESTED",
    "SUFFICIENT_FOR_POLICY",
]


def filter_inventory_plan(plan, stores, statuses, positive_orders_only):
    """Filter existing recommendations without recalculating the plan."""
    filtered = plan.loc[
        plan["store_id"].isin(stores)
        & plan["inventory_status"].isin(statuses)
    ].copy()

    if positive_orders_only:
        filtered = filtered.loc[
            filtered["suggested_order_units"] > 0
        ].copy()

    return filtered


def render_inventory_planner(data):
    """Display saved next-day inventory recommendations."""
    plan = data["inventory_recommendations"].copy()
    summary = data["planning_summary"]

    planning_date = plan["date"].iloc[0]
    snapshot_date = plan["inventory_snapshot_date"].iloc[0]

    st.title("Inventory Planner")
    st.caption(
        f"Planning date: {planning_date:%d %b %Y} | "
        f"Inventory snapshot: {snapshot_date:%d %b %Y}"
    )

    st.info(
        "This is a saved, illustrative inventory plan based on "
        "synthetic data and frozen next-day forecasts. "
        "Filters only display existing recommendations; "
        "they do not change forecasts or reorder quantities."
    )

    st.subheader("Saved Planning Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Store-Product Series",
        f"{len(plan):,}",
    )
    col2.metric(
        "Suggested Order Units",
        f"{plan['suggested_order_units'].sum():,.0f}",
    )
    col3.metric(
        "Series Requiring Orders",
        f"{(plan['suggested_order_units'] > 0).sum():,}",
    )
    col4.metric(
        "Current Stockout Series",
        f"{plan['stockout_flag'].sum():,}",
    )

    st.caption(
        "Totals above describe the complete saved plan, "
        "before any interactive filters are applied."
    )

    st.divider()
    st.subheader("Explore Recommendations")

    store_options = sorted(plan["store_id"].unique().tolist())
    status_options = [
        status
        for status in STATUS_ORDER
        if status in plan["inventory_status"].unique()
    ]

    left, middle = st.columns(2)

    with left:
        selected_stores = st.multiselect(
            "Filter by store",
            options=store_options,
            default=store_options,
            key="inventory_stores",
        )

    with middle:
        selected_statuses = st.multiselect(
            "Filter by inventory status",
            options=status_options,
            default=status_options,
            key="inventory_statuses",
        )

    positive_orders_only = st.checkbox(
        "Show only positive suggested orders",
        value=False,
    )

    filtered = filter_inventory_plan(
        plan,
        selected_stores,
        selected_statuses,
        positive_orders_only,
    )

    if filtered.empty:
        st.warning("No recommendations match the selected filters.")
        return

    st.subheader("Filtered Planning Metrics")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Matching Series",
        f"{len(filtered):,}",
    )
    metric2.metric(
        "Suggested Order Units",
        f"{filtered['suggested_order_units'].sum():,.0f}",
    )
    metric3.metric(
        "Series With Orders",
        f"{(filtered['suggested_order_units'] > 0).sum():,}",
    )

    left_chart, right_chart = st.columns(2)

    with left_chart:
        status_counts = (
            filtered["inventory_status"]
            .value_counts()
            .rename_axis("inventory_status")
            .reset_index(name="series_count")
        )

        status_chart = px.bar(
            status_counts,
            x="inventory_status",
            y="series_count",
            color="inventory_status",
            labels={
                "inventory_status": "Inventory Status",
                "series_count": "Store-Product Series",
            },
            title="Inventory Status Distribution",
        )
        status_chart.update_layout(
            height=420,
            showlegend=False,
            xaxis_tickangle=-30,
        )
        st.plotly_chart(status_chart, width="stretch")

    with right_chart:
        store_orders = (
            filtered.groupby("store_id", as_index=False)[
                "suggested_order_units"
            ]
            .sum()
            .sort_values(
                "suggested_order_units",
                ascending=False,
            )
        )

        orders_chart = px.bar(
            store_orders,
            x="store_id",
            y="suggested_order_units",
            labels={
                "store_id": "Store",
                "suggested_order_units": "Suggested Order Units",
            },
            title="Suggested Orders by Store",
        )
        orders_chart.update_layout(height=420)
        st.plotly_chart(orders_chart, width="stretch")

    st.subheader("Saved Inventory Recommendations")

    display_columns = [
        "store_id",
        "product_id",
        "inventory_status",
        "predicted_units",
        "closing_stock",
        "reorder_point",
        "supplier_lead_time_days",
        "lead_time_demand_units",
        "safety_stock_units",
        "target_stock_units",
        "suggested_order_units",
        "coverage_days",
    ]

    st.dataframe(
        filtered[display_columns].sort_values(
            "suggested_order_units",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Recommendations (CSV)",
        data=csv_bytes,
        file_name="retailpulse_filtered_inventory_plan.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Planning Policy and Limitations")

    with st.expander("How suggested orders were calculated"):
        st.markdown(
            """
            The saved planning pipeline applies these rules:

            - **Lead-time demand:** predicted daily units multiplied
              by supplier lead time.
            - **Safety stock:** ceiling of predicted daily units
              multiplied by safety days.
            - **Target stock:** ceiling of predicted daily units
              multiplied by lead time plus safety days.
            - **Suggested order:** maximum of zero and target stock
              minus closing stock.

            The calculations were performed previously by the
            Phase 7 pipeline. This dashboard does not recalculate them.
            """
        )

        st.write(
            {
                "planning_date": str(planning_date.date()),
                "inventory_snapshot_date": str(snapshot_date.date()),
                "forecast_model": summary["forecast_model"],
                "safety_days": summary["safety_days"],
                "open_purchase_orders_included": summary[
                    "open_purchase_orders_included"
                ],
                "data_source": summary["data_source"],
            }
        )

    st.warning(
        "Open purchase orders are excluded from this illustrative "
        "plan. Recommendations are not purchase orders and should "
        "not be used for real procurement decisions."
    )