-- DesignStream AI Database Seeds
-- Seed: 001_sample_products.sql
-- Description: Sample product data for development and testing

-- Insert sample products
INSERT INTO products (shopify_id, title, description, vendor, product_type) VALUES
('prod_001', 'Modern Sectional Sofa', 'A contemporary sectional sofa with clean lines and premium upholstery. Perfect for modern living spaces.', 'DesignStream', 'Furniture'),
('prod_002', 'Industrial Coffee Table', 'Rustic industrial coffee table with metal legs and reclaimed wood top. Adds character to any room.', 'DesignStream', 'Furniture'),
('prod_003', 'Minimalist Dining Chair', 'Sleek dining chair with ergonomic design and premium materials. Available in multiple colors.', 'DesignStream', 'Furniture'),
('prod_004', 'Contemporary Floor Lamp', 'Modern floor lamp with adjustable height and dimmable LED lighting. Perfect for reading corners.', 'DesignStream', 'Lighting'),
('prod_005', 'Scandinavian Bookshelf', 'Clean-lined bookshelf with open and closed storage compartments. Made from sustainable materials.', 'DesignStream', 'Storage'),
('prod_006', 'Bohemian Throw Pillow', 'Handwoven throw pillow with geometric patterns. Adds texture and color to any seating area.', 'DesignStream', 'Decor'),
('prod_007', 'Mid-Century Side Table', 'Classic mid-century side table with tapered legs and walnut finish. Timeless design.', 'DesignStream', 'Furniture'),
('prod_008', 'Modern Wall Art', 'Abstract wall art print on premium canvas. Available in multiple sizes and color schemes.', 'DesignStream', 'Art'),
('prod_009', 'Industrial Pendant Light', 'Vintage-inspired pendant light with exposed bulb and metal shade. Perfect for kitchen islands.', 'DesignStream', 'Lighting'),
('prod_010', 'Contemporary Rug', 'Hand-knotted area rug with modern geometric patterns. Soft underfoot and easy to clean.', 'DesignStream', 'Rugs');

-- Update timestamps
UPDATE products SET created_at = CURRENT_TIMESTAMP - INTERVAL '30 days' + (random() * INTERVAL '30 days');

