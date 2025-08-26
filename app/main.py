from fastapi import FastAPI
from app.routers.admin import router as admin_router
from app.routers.widget import router as widget_router
from app.routers.shopify import router as shopify_router

app = FastAPI()
app.include_router(admin_router, prefix="/admin")
app.include_router(widget_router, prefix="/widget")
app.include_router(shopify_router, prefix="/shopify")