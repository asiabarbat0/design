-- DesignStream AI Database Schema
-- Migration: 001_initial_schema.sql
-- Description: Initial database schema with pgvector support

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    shopify_id VARCHAR UNIQUE,
    title VARCHAR,
    description TEXT,
    vendor VARCHAR,
    product_type VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Variants table
CREATE TABLE IF NOT EXISTS variants (
    id SERIAL PRIMARY KEY,
    shopify_id VARCHAR UNIQUE,
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    title VARCHAR,
    price DECIMAL(10,2),
    inventory_quantity INTEGER DEFAULT 0,
    dimensions JSONB,
    dims_parsed BOOLEAN DEFAULT FALSE,
    embedding vector(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Images table
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    variant_id INTEGER REFERENCES variants(id) ON DELETE CASCADE,
    url VARCHAR,
    cutout_url VARCHAR,
    quality_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Render sessions table
CREATE TABLE IF NOT EXISTS render_sessions (
    id SERIAL PRIMARY KEY,
    user_photo_url VARCHAR,
    render_url VARCHAR,
    items JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Usage ledger table
CREATE TABLE IF NOT EXISTS usage_ledger (
    id SERIAL PRIMARY KEY,
    merchant_id VARCHAR,
    renders DECIMAL(10,2) DEFAULT 0.0,
    swaps DECIMAL(10,2) DEFAULT 0.0,
    period_start TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI recommendations table
CREATE TABLE IF NOT EXISTS ai_recommendations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR,
    room_embedding vector(512),
    recommended_variants JSONB,
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR UNIQUE,
    user_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analytics events table
CREATE TABLE IF NOT EXISTS analytics_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR,
    session_id VARCHAR,
    data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_products_updated_at 
    BEFORE UPDATE ON products 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_variants_updated_at 
    BEFORE UPDATE ON variants 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_products_shopify_id ON products(shopify_id);
CREATE INDEX IF NOT EXISTS idx_variants_shopify_id ON variants(shopify_id);
CREATE INDEX IF NOT EXISTS idx_variants_product_id ON variants(product_id);
CREATE INDEX IF NOT EXISTS idx_images_variant_id ON images(variant_id);
CREATE INDEX IF NOT EXISTS idx_render_sessions_created_at ON render_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_usage_ledger_merchant_id ON usage_ledger(merchant_id);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_session_id ON ai_recommendations(session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_analytics_events_created_at ON analytics_events(created_at);

-- Add vector similarity search index
CREATE INDEX IF NOT EXISTS idx_variants_embedding_cosine 
    ON variants USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);

-- Add GIN index for JSONB columns
CREATE INDEX IF NOT EXISTS idx_variants_dimensions_gin ON variants USING GIN (dimensions);
CREATE INDEX IF NOT EXISTS idx_render_sessions_items_gin ON render_sessions USING GIN (items);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_variants_gin ON ai_recommendations USING GIN (recommended_variants);
CREATE INDEX IF NOT EXISTS idx_user_sessions_data_gin ON user_sessions USING GIN (user_data);
CREATE INDEX IF NOT EXISTS idx_analytics_events_data_gin ON analytics_events USING GIN (data);

