"""Interactive historical retail performance dashboard."""

import plotly.express as px
import streamlit as st


def render_retail_performance(data):
    """Render store, category, and product-level BI summaries."""
    stores = data["store_performance"].copy()
    categories = data["category_performance"].copy()
    products = data["product_stockouts"].copy()

    st.title("Retail Performance")
    st.caption(
        "Historical synthetic retail data | "
        "1 January 2024 – 30 November 2025"
    )

    st.info(
        "These are historical aggregate BI summaries, not live "
        "inventory data or model predictions. Filters operate "
        "on saved reporting aggregates."
    )

    st.subheader("Store Performance")

    regions = sorted(stores["region"].dropna().unique().tolist())
    selected_regions = st.multiselect(
        "Filter stores by region",
        options=regions,
        default=regions,
    )

    filtered_stores = stores[
        stores["region"].isin(selected_regions)
    ].copy()

    if filtered_stores.empty:
        st.warning("No stores match the selected regions.")
    else:
        store_revenue = filtered_stores["net_revenue"].sum()
        store_units = filtered_stores["units_sold"].sum()
        store_lost_units = filtered_stores["lost_units"].sum()

        col1, col2, col3 = st.columns(3)

        col1.metric("Selected Stores", len(filtered_stores))
        col2.metric("Selected Revenue", f"{store_revenue:,.2f}")
        col3.metric(
            "Selected Lost Units",
            f"{store_lost_units:,.0f}",
        )

        left, right = st.columns(2)

        with left:
            revenue_chart = px.bar(
                filtered_stores.sort_values(
                    "net_revenue", ascending=False
                ),
                x="store_id",
                y="net_revenue",
                color="region",
                labels={
                    "store_id": "Store",
                    "net_revenue": "Net Revenue",
                    "region": "Region",
                },
                title="Revenue by Store",
            )
            revenue_chart.update_layout(height=420)
            st.plotly_chart(revenue_chart, width="stretch")

        with right:
            stockout_chart = px.bar(
                filtered_stores.sort_values(
                    "stockout_observation_rate_pct",
                    ascending=False,
                ),
                x="store_id",
                y="stockout_observation_rate_pct",
                color="region",
                labels={
                    "store_id": "Store",
                    "stockout_observation_rate_pct": (
                        "Stockout Observation Rate (%)"
                    ),
                    "region": "Region",
                },
                title="Store Stockout Rates",
            )
            stockout_chart.update_layout(height=420)
            st.plotly_chart(stockout_chart, width="stretch")

        st.caption(
            f"Selected stores sold {store_units:,.0f} units. "
            "Stockout rates are precomputed per store."
        )

        st.dataframe(
            filtered_stores.sort_values(
                "net_revenue", ascending=False
            ),
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.subheader("Category Performance")

    category_chart = px.bar(
        categories.sort_values(
            "net_revenue", ascending=False
        ),
        x="category",
        y="net_revenue",
        color="category",
        title="Revenue by Product Category",
        labels={
            "category": "Category",
            "net_revenue": "Net Revenue",
        },
    )
    category_chart.update_layout(
        height=400,
        showlegend=False,
    )
    st.plotly_chart(category_chart, width="stretch")

    st.dataframe(
        categories.sort_values(
            "net_revenue", ascending=False
        ),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Product Stockout Analysis")

    product_categories = sorted(
        products["category"].dropna().unique().tolist()
    )
    selected_category = st.selectbox(
        "Filter products by category",
        options=["All Categories", *product_categories],
    )

    filtered_products = products.copy()

    if selected_category != "All Categories":
        filtered_products = filtered_products[
            filtered_products["category"] == selected_category
        ]

    product_chart = px.bar(
        filtered_products.sort_values(
            "stockout_observation_rate_pct",
            ascending=False,
        ),
        x="product_id",
        y="stockout_observation_rate_pct",
        color="category",
        title="Product Stockout Observation Rates",
        labels={
            "product_id": "Product",
            "stockout_observation_rate_pct": (
                "Stockout Observation Rate (%)"
            ),
            "category": "Category",
        },
    )
    product_chart.update_layout(height=420)
    st.plotly_chart(product_chart, width="stretch")

    st.dataframe(
        filtered_products.sort_values(
            "stockout_observation_rate_pct",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Stockout rates describe historical observations. "
        "Lost-sales values are synthetic estimates and "
        "should not be interpreted as realized revenue."
    )