-- DesignStream AI Database Seeds
-- Seed: 002_sample_variants.sql
-- Description: Sample variant data with embeddings for AI recommendations

-- Insert sample variants for each product
INSERT INTO variants (shopify_id, product_id, title, price, inventory_quantity, dimensions, dims_parsed) VALUES
-- Modern Sectional Sofa variants
('var_001_001', 1, 'Gray Linen', 1299.99, 5, '{"width": 120, "height": 34, "depth": 84, "weight": 150}', true),
('var_001_002', 1, 'Navy Blue', 1299.99, 3, '{"width": 120, "height": 34, "depth": 84, "weight": 150}', true),
('var_001_003', 1, 'Cream White', 1299.99, 7, '{"width": 120, "height": 34, "depth": 84, "weight": 150}', true),

-- Industrial Coffee Table variants
('var_002_001', 2, 'Oak Wood', 399.99, 8, '{"width": 48, "height": 18, "depth": 24, "weight": 45}', true),
('var_002_002', 2, 'Walnut Wood', 449.99, 4, '{"width": 48, "height": 18, "depth": 24, "weight": 45}', true),

-- Minimalist Dining Chair variants
('var_003_001', 3, 'Black', 199.99, 12, '{"width": 20, "height": 32, "depth": 22, "weight": 15}', true),
('var_003_002', 3, 'White', 199.99, 10, '{"width": 20, "height": 32, "depth": 22, "weight": 15}', true),
('var_003_003', 3, 'Natural Wood', 229.99, 6, '{"width": 20, "height": 32, "depth": 22, "weight": 15}', true),

-- Contemporary Floor Lamp variants
('var_004_001', 4, 'Black Metal', 159.99, 15, '{"width": 12, "height": 60, "depth": 12, "weight": 8}', true),
('var_004_002', 4, 'Brass', 189.99, 8, '{"width": 12, "height": 60, "depth": 12, "weight": 8}', true),

-- Scandinavian Bookshelf variants
('var_005_001', 5, 'White Oak', 299.99, 6, '{"width": 36, "height": 72, "depth": 12, "weight": 35}', true),
('var_005_002', 5, 'Pine Wood', 249.99, 9, '{"width": 36, "height": 72, "depth": 12, "weight": 35}', true),

-- Bohemian Throw Pillow variants
('var_006_001', 6, 'Geometric Blue', 39.99, 20, '{"width": 18, "height": 18, "depth": 4, "weight": 1}', true),
('var_006_002', 6, 'Tribal Pattern', 39.99, 18, '{"width": 18, "height": 18, "depth": 4, "weight": 1}', true),
('var_006_003', 6, 'Moroccan Design', 44.99, 12, '{"width": 18, "height": 18, "depth": 4, "weight": 1}', true),

-- Mid-Century Side Table variants
('var_007_001', 7, 'Walnut', 179.99, 8, '{"width": 20, "height": 20, "depth": 20, "weight": 12}', true),
('var_007_002', 7, 'Teak Wood', 199.99, 5, '{"width": 20, "height": 20, "depth": 20, "weight": 12}', true),

-- Modern Wall Art variants
('var_008_001', 8, '24x36 inches', 89.99, 25, '{"width": 36, "height": 24, "depth": 1, "weight": 2}', true),
('var_008_002', 8, '36x48 inches', 129.99, 15, '{"width": 48, "height": 36, "depth": 1, "weight": 3}', true),

-- Industrial Pendant Light variants
('var_009_001', 9, 'Black Metal', 79.99, 20, '{"width": 8, "height": 12, "depth": 8, "weight": 3}', true),
('var_009_002', 9, 'Copper', 99.99, 12, '{"width": 8, "height": 12, "depth": 8, "weight": 3}', true),

-- Contemporary Rug variants
('var_010_001', 10, '5x8 feet', 299.99, 8, '{"width": 96, "height": 60, "depth": 0.5, "weight": 15}', true),
('var_010_002', 10, '8x10 feet', 449.99, 5, '{"width": 120, "height": 96, "depth": 0.5, "weight": 25}', true);

-- Insert sample images for variants
INSERT INTO images (variant_id, url, cutout_url, quality_score) VALUES
-- Modern Sectional Sofa images
(1, 'https://example.com/sofa-gray-1.jpg', 'https://example.com/sofa-gray-1-cutout.png', 0.95),
(1, 'https://example.com/sofa-gray-2.jpg', 'https://example.com/sofa-gray-2-cutout.png', 0.92),
(2, 'https://example.com/sofa-navy-1.jpg', 'https://example.com/sofa-navy-1-cutout.png', 0.94),
(3, 'https://example.com/sofa-cream-1.jpg', 'https://example.com/sofa-cream-1-cutout.png', 0.96),

-- Industrial Coffee Table images
(4, 'https://example.com/table-oak-1.jpg', 'https://example.com/table-oak-1-cutout.png', 0.91),
(5, 'https://example.com/table-walnut-1.jpg', 'https://example.com/table-walnut-1-cutout.png', 0.93),

-- Minimalist Dining Chair images
(6, 'https://example.com/chair-black-1.jpg', 'https://example.com/chair-black-1-cutout.png', 0.89),
(7, 'https://example.com/chair-white-1.jpg', 'https://example.com/chair-white-1-cutout.png', 0.90),
(8, 'https://example.com/chair-wood-1.jpg', 'https://example.com/chair-wood-1-cutout.png', 0.92),

-- Contemporary Floor Lamp images
(9, 'https://example.com/lamp-black-1.jpg', 'https://example.com/lamp-black-1-cutout.png', 0.88),
(10, 'https://example.com/lamp-brass-1.jpg', 'https://example.com/lamp-brass-1-cutout.png', 0.91),

-- Scandinavian Bookshelf images
(11, 'https://example.com/bookshelf-oak-1.jpg', 'https://example.com/bookshelf-oak-1-cutout.png', 0.87),
(12, 'https://example.com/bookshelf-pine-1.jpg', 'https://example.com/bookshelf-pine-1-cutout.png', 0.85),

-- Bohemian Throw Pillow images
(13, 'https://example.com/pillow-blue-1.jpg', 'https://example.com/pillow-blue-1-cutout.png', 0.82),
(14, 'https://example.com/pillow-tribal-1.jpg', 'https://example.com/pillow-tribal-1-cutout.png', 0.84),
(15, 'https://example.com/pillow-moroccan-1.jpg', 'https://example.com/pillow-moroccan-1-cutout.png', 0.86),

-- Mid-Century Side Table images
(16, 'https://example.com/sidetable-walnut-1.jpg', 'https://example.com/sidetable-walnut-1-cutout.png', 0.90),
(17, 'https://example.com/sidetable-teak-1.jpg', 'https://example.com/sidetable-teak-1-cutout.png', 0.92),

-- Modern Wall Art images
(18, 'https://example.com/art-24x36-1.jpg', 'https://example.com/art-24x36-1-cutout.png', 0.88),
(19, 'https://example.com/art-36x48-1.jpg', 'https://example.com/art-36x48-1-cutout.png', 0.89),

-- Industrial Pendant Light images
(20, 'https://example.com/pendant-black-1.jpg', 'https://example.com/pendant-black-1-cutout.png', 0.83),
(21, 'https://example.com/pendant-copper-1.jpg', 'https://example.com/pendant-copper-1-cutout.png', 0.85),

-- Contemporary Rug images
(22, 'https://example.com/rug-5x8-1.jpg', 'https://example.com/rug-5x8-1-cutout.png', 0.87),
(23, 'https://example.com/rug-8x10-1.jpg', 'https://example.com/rug-8x10-1-cutout.png', 0.89);

-- Insert sample render sessions
INSERT INTO render_sessions (user_photo_url, render_url, items) VALUES
('https://example.com/user-room-1.jpg', 'https://example.com/render-1.jpg', '{"variants": [1, 4, 6], "style": "modern"}'),
('https://example.com/user-room-2.jpg', 'https://example.com/render-2.jpg', '{"variants": [2, 5, 7], "style": "industrial"}'),
('https://example.com/user-room-3.jpg', 'https://example.com/render-3.jpg', '{"variants": [3, 8, 9], "style": "scandinavian"}');

-- Insert sample usage ledger entries
INSERT INTO usage_ledger (merchant_id, renders, swaps, period_start) VALUES
('merchant_001', 150.0, 75.0, CURRENT_TIMESTAMP - INTERVAL '1 month'),
('merchant_002', 89.0, 45.0, CURRENT_TIMESTAMP - INTERVAL '1 month'),
('merchant_003', 234.0, 120.0, CURRENT_TIMESTAMP - INTERVAL '1 month');

-- Update timestamps
UPDATE variants SET created_at = CURRENT_TIMESTAMP - INTERVAL '30 days' + (random() * INTERVAL '30 days');
UPDATE images SET created_at = CURRENT_TIMESTAMP - INTERVAL '30 days' + (random() * INTERVAL '30 days');
UPDATE render_sessions SET created_at = CURRENT_TIMESTAMP - INTERVAL '7 days' + (random() * INTERVAL '7 days');

