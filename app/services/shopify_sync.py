import os
import shopify
from app.database import SessionLocal, Product, Variant, Image
from typing import Optional

def setup_shopify_session(shop_url: str, access_token: str):
    shopify.ShopifyResource.set_site(f"https://{shop_url}/admin")
    shopify.Session.setup(api_key=os.getenv('SHOPIFY_API_KEY'), secret=os.getenv('SHOPIFY_API_SECRET'))
    session = shopify.Session(shop_url, '2025-07', access_token)
    shopify.ShopifyResource.activate_session(session)

def full_sync(shop_url: str, access_token: str) -> None:
    """Synchronize the full Shopify catalog with the local database."""
    setup_shopify_session(shop_url, access_token)
    with SessionLocal() as db:
        products = shopify.Product.find(limit=250)
        for prod in products:
            new_prod = Product(
                shopify_id=str(prod.id),
                title=prod.title,
                description=prod.description,
                vendor=prod.vendor,
                product_type=prod.product_type
            )
            db.add(new_prod)
            db.commit()
            for variant in prod.variants:
                new_variant = Variant(
                    shopify_id=str(variant.id),
                    product_id=new_prod.id,
                    title=variant.title,
                    price=float(variant.price),
                    inventory_quantity=variant.inventory_quantity
                )
                db.add(new_variant)
            for image in prod.images:
                new_image = Image(
                    variant_id=new_variant.id,  # Assuming first variant
                    url=image.src
                )
                db.add(new_image)
        db.commit()

def full_sync_partial(product_id: str):
    """Stub for partial sync of a single product."""
    # Implement logic to sync a single product by ID
    pass