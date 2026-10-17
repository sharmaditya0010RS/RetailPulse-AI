-- RETAILPULSE AI 2.0
-- PHASE 4: BUSINESS INTELLIGENCE REPORTING VIEWS
-- DATA SOURCE: SYNTHETIC RETAIL DATA
--
-- Definitions:
-- Stockout observation rate = stockout observations / all observations
-- Unmet demand rate = lost units / latent demand units
-- Estimated gross margin = estimated gross profit / net revenue
--
-- Gross profit is an estimate, not audited operating profit.
-- Lost-sales value is a scenario estimate, not realized revenue.

BEGIN;

CREATE OR REPLACE VIEW retailpulse.bi_executive_kpis AS
SELECT
    MIN(date) AS date_start,
    MAX(date) AS date_end,
    COUNT(*) AS observations,
    COUNT(DISTINCT store_id) AS stores,
    COUNT(DISTINCT product_id) AS products,
    COUNT(DISTINCT date) AS days,
    SUM(units_sold) AS units_sold,
    SUM(units_returned) AS units_returned,
    SUM(lost_sales_units) AS lost_units,
    ROUND(SUM(net_revenue), 2) AS net_revenue,
    ROUND(SUM(gross_profit_estimate), 2)
        AS estimated_gross_profit,
    ROUND(
        100.0 * SUM(gross_profit_estimate)
        / NULLIF(SUM(net_revenue), 0),
        2
    ) AS estimated_gross_margin_pct,
    ROUND(
        100.0 * SUM(stockout_flag)
        / NULLIF(COUNT(*), 0),
        2
    ) AS stockout_observation_rate_pct,
    ROUND(
        100.0 * SUM(lost_sales_units)
        / NULLIF(SUM(latent_demand_units), 0),
        2
    ) AS unmet_demand_rate_pct
FROM retailpulse.fact_daily_retail;


CREATE OR REPLACE VIEW retailpulse.bi_monthly_performance AS
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', date)::DATE AS sales_month,
        COUNT(*) AS observations,
        SUM(units_sold) AS units_sold,
        SUM(lost_sales_units) AS lost_units,
        SUM(net_revenue) AS net_revenue,
        SUM(gross_profit_estimate) AS estimated_gross_profit,
        SUM(stockout_flag) AS stockout_observations,
        SUM(latent_demand_units) AS latent_demand_units
    FROM retailpulse.fact_daily_retail
    GROUP BY 1
)
SELECT
    sales_month,
    observations,
    units_sold,
    lost_units,
    ROUND(net_revenue, 2) AS net_revenue,
    ROUND(estimated_gross_profit, 2)
        AS estimated_gross_profit,
    ROUND(
        100.0 * estimated_gross_profit
        / NULLIF(net_revenue, 0),
        2
    ) AS estimated_gross_margin_pct,
    ROUND(
        100.0 * stockout_observations
        / NULLIF(observations, 0),
        2
    ) AS stockout_observation_rate_pct,
    ROUND(
        100.0 * lost_units
        / NULLIF(latent_demand_units, 0),
        2
    ) AS unmet_demand_rate_pct,
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
    ) AS revenue_growth_mom_pct
FROM monthly;


CREATE OR REPLACE VIEW retailpulse.bi_category_performance AS
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
        / NULLIF(SUM(SUM(f.net_revenue)) OVER (), 0),
        2
    ) AS revenue_share_pct,
    ROUND(
        100.0 * SUM(f.stockout_flag)
        / NULLIF(COUNT(*), 0),
        2
    ) AS stockout_observation_rate_pct
FROM retailpulse.fact_daily_retail f
JOIN retailpulse.dim_product p
    ON f.product_id = p.product_id
GROUP BY p.category;


CREATE OR REPLACE VIEW retailpulse.bi_store_performance AS
WITH store_metrics AS (
    SELECT
        s.store_id,
        s.region,
        s.store_tier,
        SUM(f.units_sold) AS units_sold,
        SUM(f.lost_sales_units) AS lost_units,
        SUM(f.net_revenue) AS net_revenue,
        SUM(f.gross_profit_estimate)
            AS estimated_gross_profit,
        ROUND(
            100.0 * SUM(f.stockout_flag)
            / NULLIF(COUNT(*), 0),
            2
        ) AS stockout_observation_rate_pct
    FROM retailpulse.fact_daily_retail f
    JOIN retailpulse.dim_store s
        ON f.store_id = s.store_id
    GROUP BY s.store_id, s.region, s.store_tier
)
SELECT
    DENSE_RANK() OVER (
        ORDER BY net_revenue DESC
    ) AS revenue_rank,
    store_id,
    region,
    store_tier,
    units_sold,
    lost_units,
    ROUND(net_revenue, 2) AS net_revenue,
    ROUND(estimated_gross_profit, 2)
        AS estimated_gross_profit,
    stockout_observation_rate_pct
FROM store_metrics;


CREATE OR REPLACE VIEW retailpulse.bi_product_stockouts AS
SELECT
    f.product_id,
    p.category,
    p.brand,
    p.supplier_id,
    SUM(f.units_sold) AS units_sold,
    SUM(f.lost_sales_units) AS lost_units,
    SUM(f.stockout_flag) AS stockout_observations,
    ROUND(
        100.0 * SUM(f.stockout_flag)
        / NULLIF(COUNT(*), 0),
        2
    ) AS stockout_observation_rate_pct,
    ROUND(
        100.0 * SUM(f.lost_sales_units)
        / NULLIF(SUM(f.latent_demand_units), 0),
        2
    ) AS unmet_demand_rate_pct,
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
    p.brand,
    p.supplier_id;

COMMIT;