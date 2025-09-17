# In /Users/asiabarbato/Downloads/designstreamaigrok/app/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, JSON, DateTime
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
import os
from dotenv import load_dotenv
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    raise Exception("Please install pgvector: pip install pgvector")

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://asiabarbato:asiaamstel@localhost/designstreamdb")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    shopify_id = Column(String, unique=True)
    title = Column(String)
    description = Column(String)
    vendor = Column(String)
    product_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    variants = relationship("Variant", back_populates="product")

class Variant(Base):
    __tablename__ = "variants"
    id = Column(Integer, primary_key=True, index=True)
    shopify_id = Column(String, unique=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    title = Column(String)
    price = Column(Float)
    inventory_quantity = Column(Integer)
    dimensions = Column(JSON)
    dims_parsed = Column(Boolean, default=False)
    embedding = Column(Vector(512))
    product = relationship("Product", back_populates="variants")
    images = relationship("Image", back_populates="variant")

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    variant_id = Column(Integer, ForeignKey("variants.id"))
    url = Column(String)
    cutout_url = Column(String)
    quality_score = Column(Float)
    variant = relationship("Variant", back_populates="images")

class RenderSession(Base):
    __tablename__ = "render_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_photo_url = Column(String)
    render_url = Column(String)
    items = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class UsageLedger(Base):
    __tablename__ = "usage_ledger"
    id = Column(Integer, primary_key=True, index=True)
    merchant_id = Column(String)
    renders = Column(Float, default=0.0)
    swaps = Column(Float, default=0.0)
    period_start = Column(DateTime)

# Only create tables if database is available
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"[database] Could not create tables: {e}")
    print("[database] Database connection will be handled at runtime")