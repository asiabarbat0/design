-- DesignStream AI Database Schema
-- Migration: 002_add_indexes.sql
-- Description: Additional indexes for performance optimization

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_variants_product_price ON variants(product_id, price);
CREATE INDEX IF NOT EXISTS idx_variants_inventory ON variants(inventory_quantity) WHERE inventory_quantity > 0;
CREATE INDEX IF NOT EXISTS idx_images_quality ON images(quality_score) WHERE quality_score IS NOT NULL;

-- Partial indexes for active records
CREATE INDEX IF NOT EXISTS idx_products_active ON products(id) WHERE shopify_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_variants_active ON variants(id) WHERE shopify_id IS NOT NULL AND inventory_quantity > 0;

-- Text search indexes
CREATE INDEX IF NOT EXISTS idx_products_title_gin ON products USING GIN (to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_products_description_gin ON products USING GIN (to_tsvector('english', description));
CREATE INDEX IF NOT EXISTS idx_variants_title_gin ON variants USING GIN (to_tsvector('english', title));

-- Time-based partitioning indexes
CREATE INDEX IF NOT EXISTS idx_render_sessions_date ON render_sessions(DATE(created_at));
CREATE INDEX IF NOT EXISTS idx_analytics_events_date ON analytics_events(DATE(created_at));

-- Vector similarity search optimization
CREATE INDEX IF NOT EXISTS idx_variants_embedding_l2 
    ON variants USING ivfflat (embedding vector_l2_ops) 
    WITH (lists = 100);

-- Covering indexes for common queries
CREATE INDEX IF NOT EXISTS idx_variants_covering 
    ON variants(product_id, id, title, price, inventory_quantity, dims_parsed);

-- Foreign key performance indexes
CREATE INDEX IF NOT EXISTS idx_images_variant_quality ON images(variant_id, quality_score);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_confidence ON ai_recommendations(confidence_score) WHERE confidence_score > 0.5;

-- JSONB path indexes for common queries
CREATE INDEX IF NOT EXISTS idx_variants_dimensions_width ON variants USING GIN ((dimensions->>'width'));
CREATE INDEX IF NOT EXISTS idx_variants_dimensions_height ON variants USING GIN ((dimensions->>'height'));
CREATE INDEX IF NOT EXISTS idx_variants_dimensions_depth ON variants USING GIN ((dimensions->>'depth'));

-- Session-based indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_last_activity ON user_sessions(last_activity);
CREATE INDEX IF NOT EXISTS idx_analytics_events_session_type ON analytics_events(session_id, event_type);

-- Performance monitoring indexes
CREATE INDEX IF NOT EXISTS idx_usage_ledger_period ON usage_ledger(merchant_id, period_start);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_created ON ai_recommendations(created_at) WHERE confidence_score > 0.7;

