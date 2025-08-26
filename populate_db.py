# In /Users/asiabarbato/Downloads/designstreamaigrok/populate_db.py
from app.database import SessionLocal, Product, Variant
import numpy as np
from sqlalchemy.sql import text

with SessionLocal() as db:
    # Check if product exists
    product = db.execute(text("SELECT id FROM products WHERE shopify_id = :shopify_id"), {"shopify_id": "prod1"}).fetchone()
    if not product:
        product = Product(
            shopify_id="prod1",
            title="Test Product",
            description="A test product",
            vendor="Test Vendor",
            product_type="Furniture"
        )
        db.add(product)
        db.commit()
        product_id = product.id
    else:
        product_id = product[0]

    # Insert variant
    variant = Variant(
        shopify_id="test2",  # Changed to avoid duplicate variant
        product_id=product_id,
        title="Test Variant",
        price=99.99,
        inventory_quantity=10,
        dims_parsed=True,
        embedding=np.random.randn(512).astype(np.float32)
    )
    db.add(variant)
    db.commit()

    print(db.execute(text("SELECT id, shopify_id, title FROM products")).fetchall())
    print(db.execute(text("SELECT id, shopify_id, product_id, dims_parsed, inventory_quantity FROM variants")).fetchall())