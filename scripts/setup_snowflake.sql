-- =============================================================================
-- Credit Risk Intelligence Platform
-- Snowflake Database Setup Script
-- =============================================================================

-- Create database and schema
CREATE DATABASE IF NOT EXISTS CREDIT_RISK;
USE DATABASE CREDIT_RISK;

CREATE SCHEMA IF NOT EXISTS FEATURES;
CREATE SCHEMA IF NOT EXISTS RAW;
CREATE SCHEMA IF NOT EXISTS ANALYTICS;

USE SCHEMA FEATURES;

-- =============================================================================
-- Feature Store Tables
-- =============================================================================

-- Customer features table (point-in-time feature store)
CREATE TABLE IF NOT EXISTS CUSTOMER_FEATURES (
    customer_id VARCHAR(50) NOT NULL,
    feature_timestamp TIMESTAMP_NTZ NOT NULL,
    feature_version VARCHAR(10) DEFAULT 'v1',
    
    -- Temporal features
    txn_count_1h INTEGER,
    txn_count_6h INTEGER,
    txn_count_24h INTEGER,
    txn_count_7d INTEGER,
    txn_count_30d INTEGER,
    
    -- Amount features
    avg_amount_1h FLOAT,
    avg_amount_24h FLOAT,
    avg_amount_7d FLOAT,
    avg_amount_30d FLOAT,
    max_amount_7d FLOAT,
    max_amount_30d FLOAT,
    std_amount_7d FLOAT,
    std_amount_30d FLOAT,
    total_amount_30d FLOAT,
    
    -- Velocity change features
    txn_velocity_change_7d FLOAT,
    txn_velocity_change_30d FLOAT,
    amount_velocity_change_7d FLOAT,
    
    -- Volatility features
    spending_cv_7d FLOAT,
    spending_cv_30d FLOAT,
    amount_zscore FLOAT,
    
    -- Diversity features
    unique_merchants_7d INTEGER,
    unique_merchants_30d INTEGER,
    unique_categories_30d INTEGER,
    merchant_concentration_30d FLOAT,
    
    -- Temporal pattern features
    weekend_ratio_7d FLOAT,
    weekend_ratio_30d FLOAT,
    night_ratio_7d FLOAT,
    night_ratio_30d FLOAT,
    
    -- Graph features
    pagerank_score FLOAT,
    degree_centrality FLOAT,
    weighted_degree FLOAT,
    betweenness_centrality FLOAT,
    community_id INTEGER,
    community_size INTEGER,
    community_density FLOAT,
    clustering_coefficient FLOAT,
    num_merchants INTEGER,
    avg_neighbor_degree FLOAT,
    
    -- Risk propagation features
    merchant_risk_exposure FLOAT,
    high_risk_merchant_ratio FLOAT,
    max_merchant_risk FLOAT,
    weighted_avg_merchant_risk FLOAT,
    peer_risk_exposure FLOAT,
    high_risk_peer_ratio FLOAT,
    
    -- Account features
    account_age_days INTEGER,
    
    -- Metadata
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    PRIMARY KEY (customer_id, feature_timestamp)
);

-- Add clustering for performance
ALTER TABLE CUSTOMER_FEATURES CLUSTER BY (customer_id, feature_timestamp);

-- =============================================================================
-- Labels Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS LABELS (
    customer_id VARCHAR(50) NOT NULL,
    event_date DATE NOT NULL,
    is_default BOOLEAN,
    default_amount FLOAT,
    days_past_due INTEGER,
    label_type VARCHAR(20) DEFAULT 'default_30d',
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    PRIMARY KEY (customer_id, event_date)
);

-- =============================================================================
-- Model Predictions Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS MODEL_PREDICTIONS (
    prediction_id VARCHAR(50) NOT NULL,
    customer_id VARCHAR(50) NOT NULL,
    prediction_timestamp TIMESTAMP_NTZ NOT NULL,
    risk_score FLOAT NOT NULL,
    risk_level VARCHAR(10) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    feature_version VARCHAR(10),
    
    -- Explanation (top 5 features)
    top_feature_1 VARCHAR(100),
    top_feature_1_contribution FLOAT,
    top_feature_2 VARCHAR(100),
    top_feature_2_contribution FLOAT,
    top_feature_3 VARCHAR(100),
    top_feature_3_contribution FLOAT,
    top_feature_4 VARCHAR(100),
    top_feature_4_contribution FLOAT,
    top_feature_5 VARCHAR(100),
    top_feature_5_contribution FLOAT,
    
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    PRIMARY KEY (prediction_id)
);

-- Add index for customer lookups
CREATE INDEX IF NOT EXISTS idx_predictions_customer 
ON MODEL_PREDICTIONS (customer_id, prediction_timestamp);

-- =============================================================================
-- Monitoring Tables
-- =============================================================================

-- Feature drift monitoring
CREATE TABLE IF NOT EXISTS FEATURE_DRIFT_METRICS (
    metric_id VARCHAR(50) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    metric_date DATE NOT NULL,
    
    -- Statistics
    mean_value FLOAT,
    std_value FLOAT,
    min_value FLOAT,
    max_value FLOAT,
    p25 FLOAT,
    p50 FLOAT,
    p75 FLOAT,
    p99 FLOAT,
    
    -- Drift scores
    psi_score FLOAT,  -- Population Stability Index
    ks_statistic FLOAT,
    drift_detected BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    PRIMARY KEY (metric_id)
);

-- Model performance monitoring
CREATE TABLE IF NOT EXISTS MODEL_PERFORMANCE_METRICS (
    metric_id VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    metric_date DATE NOT NULL,
    
    -- Classification metrics
    auc_roc FLOAT,
    auc_pr FLOAT,
    ks_statistic FLOAT,
    gini_coefficient FLOAT,
    
    -- Threshold-based metrics
    precision_at_10 FLOAT,
    recall_at_10 FLOAT,
    f1_at_threshold FLOAT,
    
    -- Volume metrics
    total_predictions INTEGER,
    high_risk_count INTEGER,
    medium_risk_count INTEGER,
    low_risk_count INTEGER,
    
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    PRIMARY KEY (metric_id)
);

-- =============================================================================
-- Views
-- =============================================================================

-- Latest features per customer
CREATE OR REPLACE VIEW V_LATEST_CUSTOMER_FEATURES AS
SELECT *
FROM CUSTOMER_FEATURES
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY customer_id 
    ORDER BY feature_timestamp DESC
) = 1;

-- Training dataset view (point-in-time correct)
CREATE OR REPLACE VIEW V_TRAINING_DATASET AS
SELECT 
    l.customer_id,
    l.event_date,
    l.is_default,
    f.*
FROM LABELS l
INNER JOIN CUSTOMER_FEATURES f
    ON l.customer_id = f.customer_id
    AND f.feature_timestamp <= l.event_date
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY l.customer_id, l.event_date 
    ORDER BY f.feature_timestamp DESC
) = 1;

-- =============================================================================
-- Stored Procedures
-- =============================================================================

-- Procedure to compute feature statistics for monitoring
CREATE OR REPLACE PROCEDURE SP_COMPUTE_FEATURE_STATS(FEATURE_NAME VARCHAR)
RETURNS TABLE (
    feature_name VARCHAR,
    mean_value FLOAT,
    std_value FLOAT,
    min_value FLOAT,
    max_value FLOAT
)
LANGUAGE SQL
AS
$$
DECLARE
    sql_stmt VARCHAR;
BEGIN
    sql_stmt := 'SELECT ''' || FEATURE_NAME || ''' as feature_name, ' ||
                'AVG(' || FEATURE_NAME || ') as mean_value, ' ||
                'STDDEV(' || FEATURE_NAME || ') as std_value, ' ||
                'MIN(' || FEATURE_NAME || ') as min_value, ' ||
                'MAX(' || FEATURE_NAME || ') as max_value ' ||
                'FROM V_LATEST_CUSTOMER_FEATURES';
    
    RETURN TABLE(RESULTSET_FROM_STATEMENT(sql_stmt));
END;
$$;

-- =============================================================================
-- Grants (adjust roles as needed)
-- =============================================================================

-- Create roles
CREATE ROLE IF NOT EXISTS CREDIT_RISK_READER;
CREATE ROLE IF NOT EXISTS CREDIT_RISK_WRITER;
CREATE ROLE IF NOT EXISTS CREDIT_RISK_ADMIN;

-- Grant permissions
GRANT USAGE ON DATABASE CREDIT_RISK TO ROLE CREDIT_RISK_READER;
GRANT USAGE ON SCHEMA FEATURES TO ROLE CREDIT_RISK_READER;
GRANT SELECT ON ALL TABLES IN SCHEMA FEATURES TO ROLE CREDIT_RISK_READER;
GRANT SELECT ON ALL VIEWS IN SCHEMA FEATURES TO ROLE CREDIT_RISK_READER;

GRANT USAGE ON DATABASE CREDIT_RISK TO ROLE CREDIT_RISK_WRITER;
GRANT USAGE ON SCHEMA FEATURES TO ROLE CREDIT_RISK_WRITER;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA FEATURES TO ROLE CREDIT_RISK_WRITER;

GRANT ALL PRIVILEGES ON DATABASE CREDIT_RISK TO ROLE CREDIT_RISK_ADMIN;
