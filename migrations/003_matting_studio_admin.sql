-- Migration: Add Matting Studio Admin support
-- Add columns for mask data, versioning, and admin tracking

-- Add mask data storage
ALTER TABLE render_sessions 
ADD COLUMN IF NOT EXISTS mask_data TEXT,
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();

-- Add angle tags for better organization
ALTER TABLE render_sessions 
ADD COLUMN IF NOT EXISTS angle_tags TEXT[];

-- Add admin tracking
ALTER TABLE render_sessions 
ADD COLUMN IF NOT EXISTS edited_by VARCHAR(100),
ADD COLUMN IF NOT EXISTS edited_at TIMESTAMP;

-- Add version tracking
ALTER TABLE render_sessions 
ADD COLUMN IF NOT EXISTS version VARCHAR(50);

-- Add status for review workflow
ALTER TABLE render_sessions 
ADD COLUMN IF NOT EXISTS review_status VARCHAR(20) DEFAULT 'pending';

-- Create index for review queue performance
CREATE INDEX IF NOT EXISTS idx_render_sessions_review 
ON render_sessions(confidence_score, review_status, created_at) 
WHERE confidence_score < 0.7;

-- Create index for admin tracking
CREATE INDEX IF NOT EXISTS idx_render_sessions_admin 
ON render_sessions(edited_by, edited_at);

-- Update existing records to have review_status
UPDATE render_sessions 
SET review_status = CASE 
    WHEN confidence_score < 0.7 THEN 'needs_review'
    ELSE 'approved'
END
WHERE review_status IS NULL;
