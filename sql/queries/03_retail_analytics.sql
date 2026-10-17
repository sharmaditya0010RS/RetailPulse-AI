-- RETAILPULSE AI 2.0
-- PHASE 3: RETAIL ANALYTICS
-- Source: retailpulse PostgreSQL schema
-- Data: synthetic, 2024-01-01 through 2025-11-30


-- ============================================================
-- 01. EXECUTIVE KPI SUMMARY
-- ============================================================

SELECT
    COUNT(*) AS observation_count,
    COUNT(DISTINCT store_id) AS total_stores,
    COUNT(DISTINCT product_id) AS total_products,
    COUNT(DISTINCT date) AS total_days,

    SUM(units_sold) AS total_units_sold,
    SUM(units_returned) AS total_units_returned,
    SUM(lost_sales_units) AS total_lost_units,

    ROUND(SUM(net_revenue), 2) AS total_net_revenue,
    ROUND(SUM(gross_profit_estimate), 2) AS estimated_gross_profit,

    ROUND(
        100.0 * SUM(gross_profit_estimate)
        / NULLIF(SUM(net_revenue), 0),
        2
    ) AS estimated_gross_margin_pct,

    ROUND(
        100.0 * SUM(stockout_flag) / NULLIF(COUNT(*), 0),
        2
    ) AS stockout_observation_rate_pct,

    ROUND(
        100.0 * SUM(lost_sales_units)
        / NULLIF(SUM(latent_demand_units), 0),
        2
    ) AS unmet_demand_rate_pct

FROM retailpulse.fact_daily_retail;


-- ============================================================
-- 02. MONTHLY SALES AND REVENUE TRENDS
-- ============================================================

WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', date)::DATE AS sales_month,
        SUM(units_sold) AS units_sold,
        SUM(lost_sales_units) AS lost_units,
        SUM(net_revenue) AS net_revenue,
        SUM(gross_profit_estimate) AS estimated_gross_profit
    FROM retailpulse.fact_daily_retail
    GROUP BY 1
)

SELECT
    sales_month,
    units_sold,
    lost_units,
    ROUND(net_revenue, 2) AS net_revenue,
    ROUND(estimated_gross_profit, 2) AS estimated_gross_profit,

    ROUND(
        100.0 * (
            net_revenue
            - LAG(net_revenue) OVER (ORDER BY sales_month)
        )
        / NULLIF(
            LAG(net_revenue) OVER (ORDER BY sales_month),
            0
        ),
        2
    ) AS month_over_month_revenue_growth_pct

FROM monthly_sales
ORDER BY sales_month;


-- ============================================================
-- 03. CATEGORY PERFORMANCE
-- ============================================================

SELECT
    p.category,
    COUNT(DISTINCT f.product_id) AS products,
    SUM(f.units_sold) AS units_sold,
    SUM(f.lost_sales_units) AS lost_units,

    ROUND(SUM(f.net_revenue), 2) AS net_revenue,
    ROUND(SUM(f.gross_profit_estimate), 2)
        AS estimated_gross_profit,

    ROUND(
        100.0 * SUM(f.gross_profit_estimate)
        / NULLIF(SUM(f.net_revenue), 0),
        2
    ) AS estimated_gross_margin_pct,

    ROUND(
        100.0 * SUM(f.net_revenue)
        / NULLIF(
            SUM(SUM(f.net_revenue)) OVER (),
            0
        ),
        2
    ) AS revenue_share_pct

FROM retailpulse.fact_daily_retail f
JOIN retailpulse.dim_product p
    ON f.product_id = p.product_id

GROUP BY p.category
ORDER BY net_revenue DESC;


-- ============================================================
-- 04. STORE PERFORMANCE RANKINGS
-- ============================================================

WITH store_performance AS (
    SELECT
        s.store_id,
        s.region,
        s.store_tier,

        SUM(f.units_sold) AS units_sold,
        SUM(f.net_revenue) AS net_revenue,
        SUM(f.gross_profit_estimate) AS estimated_gross_profit,

        ROUND(
            100.0 * SUM(f.stockout_flag)
            / NULLIF(COUNT(*), 0),
            2
        ) AS stockout_rate_pct

    FROM retailpulse.fact_daily_retail f
    JOIN retailpulse.dim_store s
        ON f.store_id = s.store_id

    GROUP BY
        s.store_id,
        s.region,
        s.store_tier
)

SELECT
    DENSE_RANK() OVER (
        ORDER BY net_revenue DESC
    ) AS revenue_rank,

    store_id,
    region,
    store_tier,
    units_sold,
    ROUND(net_revenue, 2) AS net_revenue,
    ROUND(estimated_gross_profit, 2)
        AS estimated_gross_profit,
    stockout_rate_pct

FROM store_performance
ORDER BY revenue_rank, store_id;


-- ============================================================
-- 05. PROMOTION PERFORMANCE
-- Descriptive comparison, not causal promotion uplift.
-- ============================================================

SELECT
    CASE
        WHEN promotion_flag = 1 THEN 'Promotion'
        ELSE 'No Promotion'
    END AS promotion_group,

    COUNT(*) AS observations,

    ROUND(
        AVG(units_sold),
        2
    ) AS avg_units_sold_per_observation,

    ROUND(
        AVG(latent_demand_units),
        2
    ) AS avg_latent_demand_per_observation,

    ROUND(
        AVG(net_revenue),
        2
    ) AS avg_net_revenue_per_observation,

    ROUND(
        AVG(discount_pct) * 100,
        2
    ) AS avg_discount_pct,

    ROUND(
        100.0 * AVG(stockout_flag),
        2
    ) AS stockout_rate_pct

FROM retailpulse.fact_daily_retail
GROUP BY promotion_flag
ORDER BY promotion_flag DESC;


-- ============================================================
-- 06. PRODUCT STOCKOUT AND LOST-SALES ANALYSIS
-- Estimated lost revenue is a scenario approximation.
-- ============================================================

SELECT
    f.product_id,
    p.category,
    p.brand,

    SUM(f.units_sold) AS units_sold,
    SUM(f.lost_sales_units) AS lost_units,

    SUM(f.stockout_flag) AS stockout_days,

    ROUND(
        100.0 * SUM(f.stockout_flag)
        / NULLIF(COUNT(*), 0),
        2
    ) AS stockout_rate_pct,

    ROUND(
        SUM(f.lost_sales_units * f.selling_price),
        2
    ) AS estimated_gross_lost_sales_value

FROM retailpulse.fact_daily_retail f
JOIN retailpulse.dim_product p
    ON f.product_id = p.product_id

GROUP BY
    f.product_id,
    p.category,
    p.brand

ORDER BY lost_units DESC;


-- ============================================================
-- 07. SUPPLIER PERFORMANCE
-- Lead times are simulated supplier-related attributes.
-- Stockouts also depend on demand and inventory policy.
-- ============================================================

SELECT
    s.supplier_id,

    COUNT(DISTINCT f.product_id) AS products_supplied,

    ROUND(
        AVG(f.supplier_lead_time_days),
        2
    ) AS avg_lead_time_days,

    ROUND(
        AVG(f.supplier_reliability) * 100,
        2
    ) AS avg_supplier_reliability_pct,

    SUM(f.units_ordered) AS total_units_ordered,
    SUM(f.units_received) AS total_units_received,

    SUM(f.lost_sales_units) AS lost_sales_units,

    ROUND(
        100.0 * SUM(f.stockout_flag)
        / NULLIF(COUNT(*), 0),
        2
    ) AS stockout_rate_pct

FROM retailpulse.fact_daily_retail f
JOIN retailpulse.dim_product p
    ON f.product_id = p.product_id
JOIN retailpulse.dim_supplier s
    ON p.supplier_id = s.supplier_id

GROUP BY s.supplier_id
ORDER BY lost_sales_units DESC;