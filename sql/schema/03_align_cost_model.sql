BEGIN;

ALTER TABLE retailpulse.fact_daily_retail
    ADD COLUMN IF NOT EXISTS unit_cost NUMERIC(12, 2);

ALTER TABLE retailpulse.dim_product
    DROP COLUMN IF EXISTS unit_cost;

ALTER TABLE retailpulse.fact_daily_retail
    ADD CONSTRAINT chk_fact_unit_cost_nonnegative
    CHECK (unit_cost >= 0);

COMMIT;