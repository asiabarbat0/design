from fastapi import APIRouter, Request
import os
import shopify
from app.database import SessionLocal, Product, Variant, Image
from app.services.shopify_sync import setup_shopify_session, full_sync, full_sync_partial  # Assuming sync logic is in services
from typing import Optional

router = APIRouter()

@router.post("/webhooks/products/update")
async def product_update(request: Request):
    """Handle Shopify product update webhook."""
    data = await request.json()
    product_id = data.get("id")
    if product_id:
        full_sync_partial(product_id)  # Stub implementation
    return {"status": "accepted"}

# Optional: Add a sync endpoint for manual triggering
@router.post("/sync")
async def manual_sync(shop_url: str, access_token: str):
    """Manually trigger a full sync of the Shopify catalog."""
    full_sync(shop_url, access_token)
    return {"status": "sync initiated"}