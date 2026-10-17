CREATE SCHEMA IF NOT EXISTS retailpulse;

CREATE TABLE IF NOT EXISTS retailpulse.dim_store (
    store_id VARCHAR(10) PRIMARY KEY,
    region VARCHAR(30) NOT NULL,
    store_tier VARCHAR(30) NOT NULL,
    store_area_sqm INTEGER NOT NULL CHECK (store_area_sqm > 0)
);

CREATE TABLE IF NOT EXISTS retailpulse.dim_supplier (
    supplier_id VARCHAR(10) PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS retailpulse.dim_product (
    product_id VARCHAR(10) PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    brand VARCHAR(50) NOT NULL,
    supplier_id VARCHAR(10) NOT NULL
        REFERENCES retailpulse.dim_supplier(supplier_id),
    unit_cost NUMERIC(12, 2) NOT NULL CHECK (unit_cost >= 0)
);

CREATE TABLE IF NOT EXISTS retailpulse.fact_daily_retail (
    date DATE NOT NULL,
    store_id VARCHAR(10) NOT NULL
        REFERENCES retailpulse.dim_store(store_id),
    product_id VARCHAR(10) NOT NULL
        REFERENCES retailpulse.dim_product(product_id),

    base_price NUMERIC(12, 2) NOT NULL,
    selling_price NUMERIC(12, 2) NOT NULL,
    competitor_price NUMERIC(12, 2) NOT NULL,
    promotion_flag INTEGER NOT NULL CHECK (promotion_flag IN (0, 1)),
    promotion_type VARCHAR(30) NOT NULL,
    discount_pct NUMERIC(8, 4) NOT NULL,
    advertising_spend NUMERIC(14, 2) NOT NULL,

    holiday_flag INTEGER NOT NULL,
    festival_flag INTEGER NOT NULL,
    weekend_flag INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    month INTEGER NOT NULL,
    iso_week INTEGER NOT NULL,
    day_of_year INTEGER NOT NULL,

    temperature_c NUMERIC(8, 2) NOT NULL,
    rainfall_mm NUMERIC(10, 2) NOT NULL,
    humidity_pct NUMERIC(8, 2) NOT NULL,
    store_footfall INTEGER NOT NULL,
    staff_count INTEGER NOT NULL,
    operating_hours NUMERIC(6, 2) NOT NULL,

    loyalty_customer_share NUMERIC(8, 4) NOT NULL,
    repeat_customer_share NUMERIC(8, 4) NOT NULL,
    average_basket_size NUMERIC(8, 2) NOT NULL,
    local_demand_index NUMERIC(10, 4) NOT NULL,
    inflation_index NUMERIC(10, 5) NOT NULL,
    competitor_activity_index NUMERIC(10, 4) NOT NULL,

    opening_stock INTEGER NOT NULL CHECK (opening_stock >= 0),
    units_sold INTEGER NOT NULL CHECK (units_sold >= 0),
    units_returned INTEGER NOT NULL CHECK (units_returned >= 0),
    closing_stock INTEGER NOT NULL CHECK (closing_stock >= 0),
    lost_sales_units INTEGER NOT NULL CHECK (lost_sales_units >= 0),
    stockout_flag INTEGER NOT NULL CHECK (stockout_flag IN (0, 1)),
    reorder_point INTEGER NOT NULL,
    units_ordered INTEGER NOT NULL,
    units_received INTEGER NOT NULL,
    supplier_lead_time_days INTEGER NOT NULL,
    supplier_reliability NUMERIC(8, 4) NOT NULL,

    gross_revenue NUMERIC(16, 2) NOT NULL,
    refund_amount NUMERIC(16, 2) NOT NULL,
    net_revenue NUMERIC(16, 2) NOT NULL,
    gross_profit_estimate NUMERIC(16, 2) NOT NULL,
    latent_demand_units INTEGER NOT NULL,

    CONSTRAINT pk_fact_daily_retail
        PRIMARY KEY (date, store_id, product_id),

    CONSTRAINT chk_inventory_balance
        CHECK (closing_stock = opening_stock - units_sold),

    CONSTRAINT chk_demand_balance
        CHECK (latent_demand_units = units_sold + lost_sales_units),

    CONSTRAINT chk_returns
        CHECK (units_returned <= units_sold)
);

CREATE INDEX IF NOT EXISTS idx_fact_store_date
    ON retailpulse.fact_daily_retail(store_id, date);

CREATE INDEX IF NOT EXISTS idx_fact_product_date
    ON retailpulse.fact_daily_retail(product_id, date);