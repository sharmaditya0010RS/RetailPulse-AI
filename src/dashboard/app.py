"""RetailPulse AI 2.0 — read-only Streamlit dashboard."""

import plotly.express as px
import streamlit as st

from src.dashboard.data_loader import load_dashboard_data
from src.dashboard.forecast_performance import render_forecast_performance
from src.dashboard.inventory_planner import render_inventory_planner
from src.dashboard.retail_performance import render_retail_performance
from src.dashboard.styles import apply_enterprise_styles

st.set_page_config(
    page_title="RetailPulse AI 2.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_enterprise_styles()


@st.cache_data(show_spinner="Loading saved analytics artifacts...")
def get_dashboard_data():
    """Cache read-only artifact loading for dashboard sessions."""
    return load_dashboard_data()


def format_compact_number(value):
    """Format large values without assuming a currency."""
    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}K"
    return f"{value:,.0f}"


def render_executive_overview(data):
    """Display historical synthetic retail performance."""
    kpis = data["executive_kpis"].iloc[0]
    monthly = data["monthly_performance"].sort_values("sales_month")
    stores = data["store_performance"].sort_values(
        "net_revenue", ascending=False
    )
    categories = data["category_performance"].sort_values(
        "net_revenue", ascending=False
    )

    st.title("Executive Overview")
    st.caption(
        "Historical synthetic retail performance | "
        f"{kpis['date_start']:%d %b %Y} – "
        f"{kpis['date_end']:%d %b %Y}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Net Revenue",
        format_compact_number(kpis["net_revenue"]),
    )
    col2.metric(
        "Units Sold",
        format_compact_number(kpis["units_sold"]),
    )
    col3.metric(
        "Estimated Gross Margin",
        f"{kpis['estimated_gross_margin_pct']:.2f}%",
    )
    col4.metric(
        "Stockout Observation Rate",
        f"{kpis['stockout_observation_rate_pct']:.2f}%",
    )

    st.info(
        "All figures are generated from synthetic data. "
        "Revenue is shown in dataset monetary units; "
        "no real-world currency is assumed."
    )

    st.subheader("Monthly Revenue Trend")

    monthly_chart = px.line(
        monthly,
        x="sales_month",
        y="net_revenue",
        markers=True,
        labels={
            "sales_month": "Month",
            "net_revenue": "Net Revenue (dataset units)",
        },
    )
    monthly_chart.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
    )

    st.plotly_chart(monthly_chart, width="stretch")

    left, right = st.columns(2)

    with left:
        st.subheader("Revenue by Store")

        store_chart = px.bar(
            stores,
            x="store_id",
            y="net_revenue",
            color="region",
            labels={
                "store_id": "Store",
                "net_revenue": "Net Revenue",
                "region": "Region",
            },
        )
        store_chart.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=20, b=10),
        )

        st.plotly_chart(store_chart, width="stretch")

    with right:
        st.subheader("Revenue by Category")

        category_chart = px.bar(
            categories,
            x="category",
            y="net_revenue",
            color="category",
            labels={
                "category": "Category",
                "net_revenue": "Net Revenue",
            },
        )
        category_chart.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
        )

        st.plotly_chart(category_chart, width="stretch")

    st.subheader("Monthly Performance Data")

    display_columns = [
        "sales_month",
        "net_revenue",
        "units_sold",
        "lost_units",
        "estimated_gross_margin_pct",
        "stockout_observation_rate_pct",
        "revenue_growth_mom_pct",
    ]

    st.dataframe(
        monthly[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "sales_month": st.column_config.DateColumn(
                "Month", format="MMM YYYY"
            ),
            "net_revenue": st.column_config.NumberColumn(
                "Net Revenue", format="%.2f"
            ),
            "estimated_gross_margin_pct": (
                st.column_config.NumberColumn(
                    "Gross Margin (%)", format="%.2f"
                )
            ),
            "stockout_observation_rate_pct": (
                st.column_config.NumberColumn(
                    "Stockout Rate (%)", format="%.2f"
                )
            ),
            "revenue_growth_mom_pct": (
                st.column_config.NumberColumn(
                    "MoM Growth (%)", format="%.2f"
                )
            ),
        },
    )


def main():
    """Run the dashboard using existing immutable outputs."""
    st.sidebar.title("RetailPulse AI 2.0")
    st.sidebar.caption("Synthetic Retail Intelligence")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Executive Overview",
            "Retail Performance",
            "Forecast Performance",
            "Inventory Planner",
        ],
    )

    try:
        data = get_dashboard_data()
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Unable to load dashboard artifacts: {exc}")
        st.stop()

    if page == "Executive Overview":
        render_executive_overview(data)
    elif page == "Retail Performance":
        render_retail_performance(data)
    elif page == "Forecast Performance":
        render_forecast_performance(data)
    elif page == "Inventory Planner":
        render_inventory_planner(data)
    else:
        st.title(page)
        st.info(
            "This dashboard section will be implemented "
            "in the next Phase 8 checkpoint."
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Read-only application | Frozen forecasting champion"
    )
    st.sidebar.caption(
        "Synthetic data | Historical demonstration"
    )


if __name__ == "__main__":
    main()